import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from utils.timer import Timer
from IPython import display

def get_dataloader_workers():
    """使用4个进程读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载fashion mnist，然后加载到内存中"""
    print(f'batch_size={batch_size}')
    trains = transforms.ToTensor()
    if resize:
        trains.insert(0, transforms.Resize(resize))
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trains, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trains, download=True
    )
    print(f'len1(mnist_train)={len(mnist_train)}, len(mnist_test)={len(mnist_test)}')
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test, batch_size, shuffle=False,
                                num_workers=get_dataloader_workers()))

def get_fashion_mnist_labels(labels):
    """返回Fashion MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图像张量
            ax.imshow(img.numpy())
        else:
            # PIL图像
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes