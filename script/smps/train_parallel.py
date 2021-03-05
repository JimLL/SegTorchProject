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
DEFAULT_NUM_WORKERS_PER_GPU = 2
EPOCHS=100

def worker(rank, args):
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b6", in_channels=4, classes=10)

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
    LabelMultiChan=False
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
    # train
    model = model.train()
    for epoch in range(EPOCHS):
        if args.distributed:
            dist_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            with autocast():
                y = model(images)
                loss = Loss(y, labels.to(y.device))
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / STEPS)

        if args.distributed:
            print(
                f"[{os.getpid()}] Epoch-{epoch} ended {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD} on {y.device}"
            )
        else:
            print(f"[{os.getpid()}] Epoch-{epoch} ended on {y.device}")

    if args.distributed:
        print(
            f"[{os.getpid()}] Finishing {rank}/{DIST_DEFAULT_WORLD_SIZE} at {DIST_DEFAULT_INIT_METHOD} on {y.device}"
        )
        dist.destroy_process_group()

def launch(args):
    tic = time.time()
    if args.distributed:
        mp.spawn(worker,
                 args=(args, ),
                 nprocs=DIST_DEFAULT_WORLD_SIZE,
                 join=True)
    else:
        worker(None, args)
    toc = time.time()
    print(f"Finished in {toc-tic:.2f}s")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--distributed", action="store_true")
    args = parser.parse_args()
    launch(args)