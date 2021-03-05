#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 10
# @Author  : Luo jin
# @User    : jim 
# @File    : bn_helper.py
# -----------------------------------------
#
import torch
import functools
import torch.nn as nn

if torch.__version__.startswith('0'):
    from .sync_bn.inplace_abn.bn import InPlaceABNSync

    BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
    BatchNorm2d_class = InPlaceABNSync
    relu_inplace = False
    print("pytorch 0.x")
else:
    print("pytorch 1.x")
    # BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
    BatchNorm2d_class = BatchNorm2d = nn.BatchNorm2d
    relu_inplace = True
