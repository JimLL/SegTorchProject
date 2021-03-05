#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 13
# @Author  : Luo jin
# @User    : jim 
# @File    : infer.py
# -----------------------------------------
#
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor

from seg.models.networks.nets.unet import UNet

if __name__ == '__main__':
    # dataPath="F:\Datasets\Others\suichang_round1_test_partA_210120"
    # savePath="F:\\Datasets\\Others\\results"
    dataPath="/media/jim/DISK-F/Datasets/Others/suichang_round1_test_partA_210120"
    savePath="/media/jim/DISK-F/Datasets/Others/results"
    imagesList=os.listdir(dataPath)

    model_ckpt = "../checkpoint/unet/20210216_152718/models/state_dict_model_e_12.pt"
    model = UNet(
        dimensions=2,
        in_channels=4,
        out_channels=10,
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()
    model.cuda()
    # ----------------------------------------------------------------------------------------------
    for imgName in imagesList:
        imgPath=os.path.join(dataPath,imgName)
        image = np.array(Image.open(imgPath), dtype=np.float32)
        img_clone = image.copy()
        image = image / 255.0  # norm to 0 -1
        image = ToTensor()(image)
        image = image.unsqueeze(0)

        predictions = model(image.cuda())
        predictions = nn.Softmax(dim=1)(predictions)
        predictions = predictions.squeeze(0).cpu().detach().numpy()

        result = np.argmax(predictions, axis=0) + 1
        result_labelFormat = Image.fromarray(np.uint8(result))
        (imgNameWE,ext)=os.path.splitext(imgName)
        imgSaveName=imgNameWE+".png"
        saveNamePath=os.path.join(savePath,imgSaveName)
        result_labelFormat.save(saveNamePath)
        print(imgName)
