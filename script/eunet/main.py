#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 16
# @Author  : Luo jin
# @User    : 22403 
# @File    : main.py
# -----------------------------------------
#
import torch
import torch.nn as nn
from seg.models.efficientunet import *
if __name__ == '__main__':
    model = get_efficientunet_b6(out_channels=10, concat_input=True, pretrained=True)
    model.encoder.stem_conv=nn.Conv2d(in_channels=4,out_channels=56,kernel_size=3,stride=2,padding=1,bias=False)
    # Note: first stem_conv out_channels should be modified according to b0 ~ b7
    for n in model.named_modules():
        print(n)

    img=torch.ones((1,4,256,256))
    output=model(img)
    print(output.shape)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    flag = torch.cuda.is_available()
    print(flag)
