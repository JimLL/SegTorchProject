#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 09
# @Author  : Luo jin
# @User    : 22403 
# @File    : dataloader.py
# -----------------------------------------
#
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class Loader(Dataset):
    def __init__(self, DataPath, Augment=True, FLIPV=0.5, FLIPH=0.5, ROT=0.5, LabelMultiChan=True):
        self.DataPath = DataPath
        self.dataList = []
        self.length = self._readTXT(self.DataPath)
        self.random_indices = np.random.permutation(self.length)

        self.TTensor = ToTensor()
        self.Aug = Augment
        self.flipv = FLIPV
        self.fliph = FLIPH
        self.rotation = ROT
        self.labelMultiChannels=LabelMultiChan

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_path, label_path = self.dataList[index]
        image = np.array(Image.open(image_path), dtype=np.float32)
        image = image / 255.0  # norm to 0 - 1
        label_np_array = np.array(Image.open(label_path), dtype=np.int64) - 1

        image, label_np_array = self._Aug(image, label_np_array)
        image = self.TTensor(image)

        if self.labelMultiChannels:
            label_index = torch.from_numpy(np.expand_dims(label_np_array, 0))
            label = torch.zeros((10, 256, 256))
            label.scatter_(0, label_index, 1).float()
        else:
            label=label_np_array

        return image, label

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                image_path, label_path = line.strip().split('\t')
                self.dataList.append((image_path, label_path))
        random.shuffle(self.dataList)
        return len(self.dataList)

    def _Aug(self, image, label):
        if self.Aug:
            if self.flipv > np.random.rand():
                image = np.flip(image, axis=0)
                label = np.flip(label, axis=0)
            if self.fliph > np.random.rand():
                image = np.flip(image, axis=1)
                label = np.flip(label, axis=1)
            if self.rotation > np.random.rand():
                K = np.random.choice(4)
                image = np.rot90(image, k=K)
                label = np.rot90(label, k=K)
            # deep copy
            return image.copy(), label.copy()
        else:
            return image, label
