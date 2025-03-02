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