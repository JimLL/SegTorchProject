#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 03 - 03
# @Author  : Luo jin
# @User    : 22403 
# @File    : test.py
# -----------------------------------------
#
import torch
import numpy as np
import matplotlib.pyplot as plt

import seg.smp as smp
from seg.metrics.mIOU import IOUMetric

model = smp.UnetPlusPlus(encoder_name="efficientnet-b7", in_channels=4, classes=10)
print(model)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
m = IOUMetric(10)
m.reset_state()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6,
                                                                 last_epoch=-1)
EPOCHS = 100
STEPS = 2000
LR = []
for epoch in range(EPOCHS):
    print("curent learning rate is ", optimizer.param_groups[0]["lr"])
    for step in range(STEPS):
        LR.append(optimizer.param_groups[0]["lr"])
        scheduler.step(epoch + step / STEPS)
LR_np = np.array(LR)
x = np.arange(0, len(LR_np), 1)
plt.plot(x, LR_np, label="lr")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
