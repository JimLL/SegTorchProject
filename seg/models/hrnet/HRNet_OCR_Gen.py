#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 12
# @Author  : Luo jin
# @User    : 22403 
# @File    : HRNet_OCR_Gen.py
# -----------------------------------------
#
import torch
import torch.nn as nn

from seg.models.hrnet.config.config import _C as config
from seg.models.hrnet.config.config import update_config
from seg.models.hrnet.hrnet_ocr import get_seg_model

def HRNet_FullModel(cfgYamlPath):
    update_config(config, cfgYamlPath)
    # update_config(config, "/home/jim/PycharmProjects/SegTorchProject/seg/models/hrnet/config/config.yaml")
    model = get_seg_model(config)
    return model
