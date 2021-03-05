
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from seg.models.networks.nets.unet import Unet

def display(display_list,label_list):
    """
    图片与标签展示
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(3, 4, 1),
    plt.title('原图')
    plt.imshow(display_list[0].astype('uint8'))
    plt.axis('off')

    plt.subplot(3, 4, 2),
    plt.title('RGB_image')
    plt.imshow(display_list[1].astype('uint8'))
    plt.axis('off')

    for i in range(10):
        plt.subplot(3, 4, i+3)
        plt.imshow(display_list[i+2], cmap='gray')
        plt.title(label_list[i])
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    model = Unet(
        dimensions=2,
        in_channels=4,
        out_channels=10,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    inputs=torch.ones((4,4,256,256))
    print(model)
    output=model(inputs)
    print(output.shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # label_name = ["耕地", "林地", "草地", "道路", "城镇建设用地",
    #               "农村建设用地", "工业用地", "构筑物", "水域", "裸地"]
    # with open('../train.txt', 'r') as f:
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
    #         label=label.numpy()
    #         # image = np.flip(image, axis=0)
    #         # label = np.flip(label, axis=0)
    #         label=np.rot90(label,k=1)
    #         image=np.rot90(image,k=1)
    #         image_RGB = image[:, :, 0:3]
    #         display_list = [image, image_RGB, label[:, :, 0], label[:, :, 1],
    #                         label[:, :, 2], label[:, :, 3], label[:, :, 4], label[:, :, 5],
    #                         label[:, :, 6], label[:, :, 7], label[:, :, 8], label[:, :, 9]]
    #
    #         display(display_list, label_name)
    #         break

