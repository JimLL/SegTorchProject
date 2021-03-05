#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 09
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
import torch.nn.functional as F
import matplotlib.pyplot as plt
from seg.data.dataloader import Loader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from seg.models.hrnet.HRNet_OCR_Gen import HRNet_FullModel
from seg.loss.dice import GeneralizedDiceLoss, GeneralizedWassersteinDiceLoss, LogCoshGeneralizedDiceLoss
from seg.loss.focal_loss import FocalLoss
from seg.loss.tversky import TverskyLoss
from seg.metrics.mIOU import IOUMetric

if __name__ == '__main__':
    EPOCHS = 50
    InteLog = 10
    schedulerStep = 15
    batch_size = 2
    # ----------------------------------------------------------------------------------------------
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_path = os.path.join('../checkpoint/hrnet', current_time)
    if not os.path.exists(current_path): os.mkdir(current_path)
    model_dir = os.path.join(current_path, 'models')
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    log_dir = os.path.join(current_path, 'logs')
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    # ----------------------------------------------------------------------------------------------
    trainTxtPath = '../train.txt'
    datasets = Loader(trainTxtPath, Augment=True)
    feeder = DataLoader(datasets, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(),
                        drop_last=True, num_workers=6)
    STEPS = len(feeder)

    valTxtPath = '../val.txt'
    datasets_val = Loader(valTxtPath, Augment=False)
    feeder_val = DataLoader(datasets_val, batch_size=1, shuffle=False, pin_memory=torch.cuda.is_available(),
                            drop_last=False, num_workers=2)
    STEPS_val = len(feeder_val)
    # ----------------------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path="E:/PycharmProjects/SegTorchProject/seg/models/hrnet/config/config.yaml"
    model = HRNet_FullModel(cfg_path)
    # model_ckpt = "../checkpoint/hrnet/20210214_094422/models/state_dict_model_e_0.pt"
    # model.load_state_dict(torch.load(model_ckpt))
    model.train()
    model.cuda()
    # ----------------------------------------------------------------------------------------------
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    # ----------------------------------------------------------------------------------------------
    GDiceLoss = GeneralizedDiceLoss(softmax=True)
    LCGDiceLoss = LogCoshGeneralizedDiceLoss(softmax=True)
    # Focal_loss = FocalLoss(gamma=8.0)
    Tversky_loss = TverskyLoss(softmax=True, alpha=0.7, beta=0.3)
    # ----------------------------------------------------------------------------------------------
    metirc = IOUMetric(num_classes=10)
    # ----------------------------------------------------------------------------------------------
    for epoch in range(EPOCHS):
        print("curent learning rate is ", optimizer.param_groups[0]["lr"])
        # ----------------------------------------------------------------------------------------------
        # train for in epoch
        for step, (images, labels) in enumerate(feeder):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            preds_down = model(images)
            h, w = labels.size(2), labels.size(3)
            preds=F.interpolate(preds_down[1],(h,w),mode="bilinear")
            # loss = GDiceLoss(preds, labels)
            # Floss = Focal_loss(preds, labels)
            loss = Tversky_loss(preds, labels)

            if step % InteLog == 0:
                pred_index = np.argmax(preds.cpu().detach().numpy(), axis=1)
                label_index = np.argmax(labels.cpu().detach().numpy(), axis=1)
                metirc.add_batch(pred_index, label_index)
                _, _, ius, mean_iu, _ = metirc.evaluate()
                loss_scalar = loss.cpu().detach().numpy()
                print('Epoch [{}][{}/{}]: loss: {:.6f}  mIOU: {:.6f}'.format(epoch + 1, step, STEPS,
                                                                                                loss_scalar, mean_iu))
                print(
                    "ious: 0:{:.3f} 1:{:.3f} 2:{:.3f} 3:{:.3f} 4:{:.3f} 5:{:.3f} 6:{:.3f} 7:{:.3f} 8:{:.3f} 9:{:.3f} ".format(
                        ius[0], ius[1], ius[2], ius[3], ius[4], ius[5], ius[6], ius[7], ius[8], ius[9]))
                writer.add_scalar('loss', loss_scalar, epoch * STEPS + step)
                writer.add_scalar('mIOU', mean_iu, epoch * STEPS + step)
                for i in range(len(ius)):
                    section = 'IOU/%d' % i
                    writer.add_scalar(section, ius[i], epoch * STEPS + step)
            loss.backward()
            optimizer.step()
        # ----------------------------------------------------------------------------------------------
        # val for in epoch
        model.eval()
        with torch.no_grad():
            loss_val = np.zeros((STEPS_val, 1))
            mIOU_val = np.zeros((STEPS_val, 1))
            IOUs_val = np.zeros((STEPS_val, 10))
            for step, (images, labels) in enumerate(feeder_val):
                images = images.cuda()
                labels = labels.cuda()
                preds_down = model(images)
                h, w = labels.size(2), labels.size(3)
                preds = F.interpolate(preds_down[1], (h, w), mode="bilinear")
                loss = Tversky_loss(preds, labels)

                pred_index = np.argmax(preds.cpu().detach().numpy(), axis=1)
                label_index = np.argmax(labels.cpu().detach().numpy(), axis=1)
                metirc.add_batch(pred_index, label_index)
                _, _, ius, mean_iu, _ = metirc.evaluate()
                loss_scalar = loss.cpu().detach().numpy()
                loss_val[step] = loss_scalar
                mIOU_val[step] = mean_iu
                IOUs_val[step, :] = ius
            val_loss_m = np.mean(loss_val)
            mIOU_m = np.mean(mIOU_val)
            ius_m = np.mean(IOUs_val, axis=0)
            print('Epoch [{}]: loss_val: {:.6f} mIOU: {:.6f}'.format(epoch + 1,val_loss_m,mIOU_m))
            print(
                "ious: 0:{:.3f} 1:{:.3f} 2:{:.3f} 3:{:.3f} 4:{:.3f} 5:{:.3f} 6:{:.3f} 7:{:.3f} 8:{:.3f} 9:{:.3f} ".format(
                    ius_m[0], ius_m[1], ius_m[2], ius_m[3], ius_m[4], ius_m[5], ius_m[6], ius_m[7], ius_m[8], ius_m[9]))
            writer.add_scalar('loss_val', val_loss_m, epoch)
            writer.add_scalar('mIOU_val', mIOU_m, epoch)
            for i in range(10):
                section = 'IOU_val/%d' % i
                writer.add_scalar(section, ius_m[i], epoch)

        # ----------------------------------------------------------------------------------------------
        if (epoch + 1) % schedulerStep == 0: scheduler.step()
        model_subdir = "state_dict_model_e_%d.pt" % epoch
        model_save_name = os.path.join(model_dir, model_subdir)
        torch.save(model.state_dict(), model_save_name)
        torch.cuda.empty_cache()  # empty cuda cache
    writer.close()

