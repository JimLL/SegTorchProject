#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021 - 02 - 15
# @Author  : Luo jin
# @User    : 22403 
# @File    : genTxt.py
# -----------------------------------------
#
import os
import random

def sort_images(image_dir, image_type):
    """
    对文件夹内的图像进行按照文件名排序
    """
    files = []

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.{}'.format(image_type)) \
                and not image_name.startswith('.'):
            files.append(os.path.join(image_dir, image_name))

    return sorted(files)

def write_file(mode, images, labels):
    """
    保存图片与对应标签至一个.txt文件
    """
    with open('./{}.txt'.format(mode), 'w') as f:
        for i in range(len(images)):
            if i==len(images)-1:
                f.write('{}\t{}'.format(images[i], labels[i]))
            else:
                f.write('{}\t{}\n'.format(images[i], labels[i]))
  # 下述代码用来写入图片标签文件名一一对映的训练集数据库
# train_images_path="/media/jim/DISK/TEMP/Datasets/Others/suichang_round1_train_210120"
train_images_path="F:\Datasets\Others\suichang_round1_train_210120"
train_images = sort_images(train_images_path, 'tif')
train_labels = sort_images(train_images_path, 'png')
val_ratio=0.1
data_nums=len(train_images)
val_data_num=int(data_nums*val_ratio)
images_for_train=[]
labels_for_train=[]
images_for_val=[]
labels_for_val=[]
for i in range(data_nums):
    prob=random.random()
    if prob>val_ratio:
        images_for_train.append(train_images[i])
        labels_for_train.append(train_labels[i])
    else:
        images_for_val.append(train_images[i])
        labels_for_val.append(train_labels[i])
write_file('train', images_for_train, labels_for_train)
write_file('val', images_for_val, labels_for_val)

