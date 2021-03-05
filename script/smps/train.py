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

if __name__ == '__main__':
    EPOCHS = 100
    InteLog = 10
    batch_size_train = 4  # for all GPUs
    batch_size_val = 1  # for all GPUs
    num_workers = 6
    LabelMultiChan = False

    trainTxtPath = 'E:/PycharmProjects/SegTorchProject/script/train.txt'
    valTxtPath = 'E:/PycharmProjects/SegTorchProject/script/val.txt'
    # ----------------------------------------------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_path = os.path.join('E:/PycharmProjects/SegTorchProject/script/checkpoint/smps', current_time)
    if not os.path.exists(current_path): os.mkdir(current_path)
    model_dir = os.path.join(current_path, 'models')
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    log_dir = os.path.join(current_path, 'logs')
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    # ----------------------------------------------------------------------------------------------
    datasets = Loader(trainTxtPath, Augment=True, LabelMultiChan=LabelMultiChan)
    feeder = DataLoader(datasets, batch_size=batch_size_train, shuffle=True,
                        pin_memory=False,
                        drop_last=True, num_workers=num_workers)
    STEPS = len(feeder)

    datasets_val = Loader(valTxtPath, Augment=True, LabelMultiChan=LabelMultiChan)
    feeder_val = DataLoader(datasets_val, batch_size=batch_size_val, shuffle=True,
                            pin_memory=False,
                            drop_last=False, num_workers=num_workers)
    STEPS_val = len(feeder_val)
    # ----------------------------------------------------------------------------------------------
    model = smp.UnetPlusPlus(encoder_name="efficientnet-b6", in_channels=4, classes=10)
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.train()
    model.cuda()
    # ----------------------------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6,
                                                                     last_epoch=-1)
    # ----------------------------------------------------------------------------------------------
    # lossf = smp.losses.LovaszLoss("multiclass").cuda() # unstable and why? IOU? non Convex optimization?
    loss_dice = smp.losses.DiceLoss(mode="multiclass")
    loss_soft_ce = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)
    Loss = smp.losses.JointLoss(loss_dice, loss_soft_ce, 1.0, 1.0)

    # following loss is designed for label:NxCxHXW (LabelMultiChan=True)
    # lovasz_softmax is better than BCE/Jaccard, it's extention of jaccard(IOU)
    # Tversky loss is extention of dice coeff + jaccard coeff(IOU)
    # Generalized Dice loss is extention of dice coeff to integrate multi class
    # BCE + Dice 在数据较为均衡的情况下有所改善,
    # 但是在数据极度不均衡的情况下交叉熵Loss会在迭代几个Epoch之后远远小于Dice Loss，这个组合Loss会退化为Dice Loss
    # LogCoshGeneralizedDiceLoss : Log-Cosh for smoothing dice
    # ----------------------------------------------------------------------------------------------
    metirc = IOUMetric(num_classes=10)
    # ----------------------------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        print("curent learning rate is ", optimizer.param_groups[0]["lr"])
        # ----------------------------------------------------------------------------------------------
        # train for in a epoch
        for step, (images, labels) in enumerate(feeder):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            with autocast():
                preds = model(images)
                loss = Loss(preds, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + step / STEPS)

            if step % InteLog == 0:
                pred_index = torch.argmax(preds, dim=1).detach().cpu().numpy()
                if LabelMultiChan:
                    label_index = torch.argmax(labels, dim=1).detach().cpu().numpy()
                else:
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
                writer.add_scalar('loss', loss_scalar, epoch * STEPS + step)
                writer.add_scalar('mIOU', mean_iu, epoch * STEPS + step)
                writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch * STEPS + step)
                for i in range(len(ius)):
                    section = 'IOU/%d' % i
                    writer.add_scalar(section, ius[i], epoch * STEPS + step)

        # ----------------------------------------------------------------------------------------------
        # val for in a epoch
        model.eval()
        with torch.no_grad():
            loss_val = np.zeros((STEPS_val, 1))
            mIOU_val = np.zeros((STEPS_val, 1))
            IOUs_val = np.zeros((STEPS_val, 10))
            for step, (images, labels) in enumerate(feeder_val):
                images = images.cuda()
                labels = labels.cuda()
                preds = model(images)
                loss_v = Loss(preds, labels)
                pred_index = torch.argmax(preds, dim=1).detach().cpu().numpy()
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
                    ius_m[0], ius_m[1], ius_m[2], ius_m[3], ius_m[4], ius_m[5], ius_m[6], ius_m[7], ius_m[8], ius_m[9]))
            writer.add_scalar('loss_val', val_loss_m, epoch)
            writer.add_scalar('mIOU_val', mIOU_m, epoch)
            for i in range(10):
                section = 'IOU_val/%d' % i
                writer.add_scalar(section, ius_m[i], epoch)
        metirc.reset_state()

        model_subdir = "state_dict_model_e_%d.pt" % (epoch + 1)
        model_save_name = os.path.join(model_dir, model_subdir)

        torch.save(model.state_dict(), model_save_name)
        torch.cuda.empty_cache()  # empty cuda cache
    writer.close()
