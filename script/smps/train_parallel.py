#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 03 - 03
# @Author  : Luo jin
# @User    : 22403 
# @File    : train.py
# -----------------------------------------
#
import os
import torch
import datetime
import numpy as np
import torch.optim as optim
from seg.data.dataloader import Loader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast

from seg.metrics.mIOU import IOUMetric
from seg.loss.dice import GeneralizedDiceLoss, LogCoshGeneralizedDiceLoss
from seg.loss.tversky import TverskyLoss
import seg.smp as smp

import time
import torch.cuda as cuda
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as data
import torch.multiprocessing as mp
from argparse import ArgumentParser

DIST_DEFAULT_BACKEND = 'nccl'
DIST_DEFAULT_ADDR = 'localhost'
DIST_DEFAULT_PORT = '12132'
DIST_DEFAULT_INIT_METHOD = f'tcp://{DIST_DEFAULT_ADDR}:{DIST_DEFAULT_PORT}'
DIST_DEFAULT_WORLD_SIZE = cuda.device_count()

DEFAULT_BATCH_SIZE = 6
DEFAULT_BATCH_SIZE_val = 2
DEFAULT_NUM_WORKERS_PER_GPU = 2
EPOCHS = 100
InteLog = 10


def worker(rank, args):
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b6", in_channels=4, classes=10)
    metirc = IOUMetric(num_classes=10)
    if args.distributed:
        print(
            f"[{os.getpid()}] Initializing {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD}"
        )
        # initialize with TCP in this example
        dist.init_process_group(backend=DIST_DEFAULT_BACKEND,
                                init_method=DIST_DEFAULT_INIT_METHOD,
                                world_size=DIST_DEFAULT_WORLD_SIZE,
                                rank=rank)

        # Another way to initialize with environment variables
        # os.environ["MASTER_PORT"] = DIST_DEFAULT_PORT
        # os.environ["MASTER_ADDR"] = DIST_DEFAULT_ADDR
        # os.environ["WORLD_SIZE"] = str(DIST_DEFAULT_WORLD_SIZE)
        # os.environ["RANK"] = str(rank)
        # dist.init_process_group(backend=DIST_DEFAULT_BACKEND)

        print(
            f"[{os.getpid()}] Computing {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD}"
        )
        # ensuring that each process exclusively works on a single GPU
        torch.cuda.set_device(rank)
        model.cuda(rank)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        model = nn.DataParallel(model).cuda()

    loss_dice = smp.losses.DiceLoss(mode="multiclass")
    loss_soft_ce = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)
    Loss = smp.losses.JointLoss(loss_dice, loss_soft_ce, 1.0, 1.0)

    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6,
                                                                     last_epoch=-1)
    # dataset
    LabelMultiChan = False
    trainTxtPath = 'E:/PycharmProjects/SegTorchProject/script/train.txt'
    dataset = Loader(trainTxtPath, Augment=True, LabelMultiChan=LabelMultiChan)
    dist_sampler = data.distributed.DistributedSampler(
        dataset) if args.distributed else None
    dataloader = data.dataloader.DataLoader(
        dataset,
        batch_size=DEFAULT_BATCH_SIZE // DIST_DEFAULT_WORLD_SIZE if args.distributed else DEFAULT_BATCH_SIZE,
        shuffle=(dist_sampler is None),
        num_workers=DEFAULT_NUM_WORKERS_PER_GPU if args.distributed else DEFAULT_NUM_WORKERS_PER_GPU * DIST_DEFAULT_WORLD_SIZE,
        sampler=dist_sampler)
    STEPS = len(dataloader)

    valTxtPath = 'E:/PycharmProjects/SegTorchProject/script/val.txt'
    datasets_val = Loader(valTxtPath, Augment=True, LabelMultiChan=LabelMultiChan)
    dist_sampler_val = data.distributed.DistributedSampler(
        datasets_val) if args.distributed else None
    dataloader_val = data.dataloader.DataLoader(
        datasets_val,
        batch_size=DEFAULT_BATCH_SIZE_val // DIST_DEFAULT_WORLD_SIZE if args.distributed else DEFAULT_BATCH_SIZE_val,
        shuffle=(dist_sampler_val is None),
        num_workers=DEFAULT_NUM_WORKERS_PER_GPU if args.distributed else DEFAULT_NUM_WORKERS_PER_GPU * DIST_DEFAULT_WORLD_SIZE,
        sampler=dist_sampler_val)
    STEPS_val = len(dataloader_val)
    # train
    for epoch in range(EPOCHS):
        model.train()
        if args.distributed:
            dist_sampler.set_epoch(epoch)
        for step, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            with autocast():
                y = model(images)
                loss = Loss(y, labels.to(y.device))
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + step / STEPS)

            if step % InteLog == 0:
                pred_index = torch.argmax(y, dim=1).detach().cpu().numpy()
                label_index = labels.detach().cpu().numpy()
                metirc.add_batch(pred_index, label_index)
                _, _, ius, mean_iu, _ = metirc.evaluate()
                metirc.reset_state()
                loss_scalar = loss.detach().cpu().numpy()
                print('Epoch [{}][{}/{}]: loss: {:.6f} mIOU: {:.6f}'.format(epoch + 1, step, STEPS,
                                                                            loss_scalar, mean_iu))
                print(
                    "ious: 0:{:.3f} 1:{:.3f} 2:{:.3f} 3:{:.3f} 4:{:.3f} 5:{:.3f} 6:{:.3f} 7:{:.3f} 8:{:.3f} 9:{:.3f} ".format(
                        ius[0], ius[1], ius[2], ius[3], ius[4], ius[5], ius[6], ius[7], ius[8], ius[9]))

        if args.distributed:
            print(
                f"[{os.getpid()}] Epoch-{epoch} ended {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD} on {y.device}"
            )
        else:
            print(f"[{os.getpid()}] Epoch-{epoch} ended on {y.device}")

        # ----------------------------------------------------------------------------------------------
        # val for in a epoch
        if args.distributed:
            dist_sampler_val.set_epoch(epoch)
        model.eval()
        with torch.no_grad():
            loss_val = np.zeros((STEPS_val, 1))
            mIOU_val = np.zeros((STEPS_val, 1))
            IOUs_val = np.zeros((STEPS_val, 10))
            for step, (images, labels) in enumerate(dataloader_val):
                y = model(images)
                loss_v = Loss(y, labels.to(y.device))

                pred_index = torch.argmax(y, dim=1).detach().cpu().numpy()
                if LabelMultiChan:
                    label_index = torch.argmax(labels, dim=1).detach().cpu().numpy()
                else:
                    label_index = labels.detach().cpu().numpy()
                metirc.add_batch(pred_index, label_index)
                _, _, ius, mean_iu, _ = metirc.evaluate()
                loss_scalar = loss_v.detach().cpu().numpy()
                loss_val[step] = loss_scalar
                mIOU_val[step] = mean_iu
                IOUs_val[step, :] = ius
            val_loss_m = np.mean(loss_val)
            mIOU_m = np.mean(mIOU_val)
            ius_m = np.mean(IOUs_val, axis=0)
            print('Epoch [{}]: loss_val: {:.6f} mIOU: {:.6f}'.format(epoch + 1, val_loss_m, mIOU_m))
            print(
                "ious: 0:{:.3f} 1:{:.3f} 2:{:.3f} 3:{:.3f} 4:{:.3f} 5:{:.3f} 6:{:.3f} 7:{:.3f} 8:{:.3f} 9:{:.3f} ".format(
                    ius_m[0], ius_m[1], ius_m[2], ius_m[3], ius_m[4], ius_m[5], ius_m[6], ius_m[7], ius_m[8],
                    ius_m[9]))
        metirc.reset_state()

    if args.distributed:
        print(
            f"[{os.getpid()}] Finishing {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD} on {y.device}"
        )
        dist.destroy_process_group()


def launch(args):
    tic = time.time()
    if args.distributed:
        mp.spawn(worker,
                 args=(args,),
                 nprocs=DIST_DEFAULT_WORLD_SIZE,
                 join=True)
    else:
        worker(None, args)
    toc = time.time()
    print(f"Finished in {toc - tic:.2f}s")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--distributed", action="store_true")
    args = parser.parse_args()
    launch(args)
