#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 09
# @Author  : Luo jin
# @User    : 22403 
# @File    : mIOU.py
# -----------------------------------------
#
import torch
import torch.nn as nn
import numpy as np

class IOUMetric(object):
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.eps=1e-5

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / (self.hist.sum()+self.eps)
        acc_cls = np.diag(self.hist) / (self.hist.sum(axis=1)++self.eps)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist)+self.eps)
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / (self.hist.sum()+self.eps)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

    def reset_state(self):
        self.hist *= 0.0


class IOUMetric_tensor(nn.Module):
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes, to_device):
        super(IOUMetric_tensor, self).__init__()
        self.num_classes = num_classes
        self.hist = torch.zeros((num_classes, num_classes)).to(to_device)
        self.eps = 1e-5

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = torch.bincount(
            self.num_classes * label_true[mask].int() +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = torch.diag(self.hist).sum() / (self.hist.sum() + self.eps)
        acc_cls = torch.diag(self.hist) / (self.hist.sum(axis=1) + +self.eps)
        acc_cls = torch.nansum(acc_cls) / len(acc_cls)
        iu = torch.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - torch.diag(self.hist) + self.eps)
        mean_iu = torch.nansum(iu) / len(iu)
        freq = self.hist.sum(axis=1) / (self.hist.sum() + self.eps)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

    def reset_state(self):
        self.hist *= 0.0