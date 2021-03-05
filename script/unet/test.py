#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 10
# @Author  : Luo jin
# @User    : 22403 
# @File    : test.py
# -----------------------------------------
#
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from script.utils import display
from seg.models.networks.nets.unet import UNet


if __name__ == '__main__':
    img_path="D:/TEMP/Datasets/Others/suichang_round1_train_210120/000027.tif"
    image = np.array(Image.open(img_path), dtype=np.float32)
    print(image.shape)
    img_clone=image.copy()
    image = image / 255.0  # norm to 0 -1
    image = ToTensor()(image)
    image=image.unsqueeze(0)
    print(image.shape)
    # ----------------------------------------------------------------------------------------------
    model_ckpt="../checkpoint/unet/20210212_221154/models/state_dict_model_e_49.pt"
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
    predictions=model(image.cuda())
    predictions=nn.Softmax(dim=1)(predictions)
    predictions=predictions.squeeze(0).cpu().detach().numpy()
    display(img_clone[:,:,0:3],predictions)
    print(predictions.shape)
    result=np.argmax(predictions,axis=0)+1
    print(result.shape)
    result_labelFormat = Image.fromarray(np.uint8(result))
    print(result_labelFormat)

