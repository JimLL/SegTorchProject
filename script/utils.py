#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 10
# @Author  : Luo jin
# @User    : jim 
# @File    : utils.py
# -----------------------------------------
#
import matplotlib.pyplot as plt

def display(srcImg,pred):
    """
    图片与标签展示
    """
    label_list=["耕地", "林地", "草地", "道路", "城镇建设用地",
                  "农村建设用地", "工业用地", "构筑物", "水域", "裸地"]
    plt.rcParams['font.sans-serif'] = ['simhei']
    # plt.rcParams['axes.unicode_minus'] = False
    plt.subplot(3, 4, 1),
    plt.title('原图')
    plt.imshow(srcImg.astype('uint8'))
    plt.axis('off')

    # plt.subplot(3, 4, 2),
    # plt.title('RGB_image')
    # plt.imshow(display_list[1].astype('uint8'))
    # plt.axis('off')

    for i in range(10):
        plt.subplot(3, 4, i+3)
        plt.imshow(pred[i,:,:], cmap='gray')
        plt.title(label_list[i])
        plt.axis('off')
    plt.show()
