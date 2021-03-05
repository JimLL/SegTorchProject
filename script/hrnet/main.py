
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from seg.models.networks.nets.unet import Unet
from seg.models.hrnet.HRNet_OCR_Gen import HRNet_FullModel
if __name__ == '__main__':
    cfg_path = "E:/PycharmProjects/SegTorchProject/seg/models/hrnet/config/config.yaml"
    model = HRNet_FullModel(cfg_path)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # label_name = ["耕地", "林地", "草地", "道路", "城镇建设用地",
    #               "农村建设用地", "工业用地", "构筑物", "水域", "裸地"]
    # with open('./script/train.txt', 'r') as f:
    #
    #     for line in f.readlines():
    #         image_path, label_path = line.strip().split('\t')
    #         label_np=np.array(Image.open(label_path),dtype=np.int64)-1
    #         label_src = torch.from_numpy(np.expand_dims(label_np,-1))
    #         label=torch.zeros((256,256,10))
    #         label.scatter_(2, label_src,1).float()
    #         print(label)
    #
    #         image=np.array(Image.open(image_path))
    #         print(image.shape)
    #         image_RGB=image[0,:,:]
    #         label=label.numpy()
    #         display_list = [image, image_RGB, label[:, :, 0], label[:, :, 1],
    #                         label[:, :, 2], label[:, :, 3], label[:, :, 4], label[:, :, 5],
    #                         label[:, :, 6], label[:, :, 7], label[:, :, 8], label[:, :, 9]]
    #
    #         display(display_list, label_name)
    #         break

