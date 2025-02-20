import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from utils.timer import Timer


def get_dataloader_workers():
    """使用4个进程读取数据"""
    return 4

def init_fashion_mnist():
    """初始化fashion mnist"""
    trains = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trains, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trains, download=True
    )
    print(len(mnist_train), len(mnist_test))
    batch_size = 256
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers())
    timer = Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

def get_fashion_mnist_labels(labels):
    """返回fashion mnist数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if (torch.is_tensor(img)):
            # 图像张量
            ax.imshow(img.numpy())
        else:
            # pil图像
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if (titles):
            ax.set_title(titles[i])
    return axes

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

if __name__ == '__main__':
    init_fashion_mnist()