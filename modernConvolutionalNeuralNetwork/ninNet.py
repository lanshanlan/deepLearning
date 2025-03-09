import torch
from torch import nn
from utils.loadData import load_data_fashion_mnist
from utils.train_ch6 import train_ch6
from utils.print_layer import print_layer
from utils.gpu import try_gpu

# NiN卷积神经网络
# 取消了全连接层，改为用一个NiN块，其输出通道数等于标签类别数，最后放一个全局平均汇聚层(global average pooling layer)
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10，输出通道等于标签类别数
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1,1)),
    # 将四维的输出转为二维输出，其形状为（批量大小，10）
    nn.Flatten()
)

def print_nin_layer():
    X_shape = (1, 1, 224, 224)
    print_layer(net, X_shape)

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

def test_ninNet():
    lr, num_epochs = 0.1, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())