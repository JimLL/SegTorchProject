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

from seg.models.efficientunet import *

if __name__ == '__main__':
    # dataPath="F:\Datasets\Others\suichang_round1_test_partA_210120"
    # savePath="F:\\Datasets\\Others\\results"
    dataPath="/media/jim/DISK-F/Datasets/Others/suichang_round1_test_partA_210120"
    savePath="/media/jim/DISK-F/Datasets/Others/results"
    imagesList=os.listdir(dataPath)

    model_ckpt = "../checkpoint/eunet/20210216_234805/models/state_dict_model_e_75.pt"
    model = get_efficientunet_b0(out_channels=10, concat_input=True, pretrained=True)
    model.encoder.stem_conv = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
    # modified stem conv for 4 channels
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
