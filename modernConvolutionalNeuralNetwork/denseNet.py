import torch
from torch import nn
from utils.loadData import load_data_fashion_mnist
from utils.train_ch6 import train_ch6
from utils.print_layer import print_layer
from utils.gpu import try_gpu
from torch.nn import functional as F

# DenseNet稠密连接网络
# 缺点是显存消耗过多
# 设计了稠密块（dense block）和过渡层（transition layer）
# 前者定义如何连接输入和输出，后者则控制通道数，使其不太复杂

# 卷积块
# 卷积块的通道数控制了输出通道数相对于输入通道数的增长速度，因此也被称为增长率（growth rate）
def conv_block(input_channels, num_channels):
    # 改良了ResNet架构，改为了"批量规范化层、激活层、卷积层"
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

# 稠密块
class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels
            ))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个卷积块的输入和输出
            X = torch.cat((X, Y), dim=1)
        return X
     
def print_Dense_shape():
    blk = DenseBlock(4,3,10)
    X = torch.rand(4,3,8,8)
    print(blk(X).shape)

# 过渡层，用来控制模型复杂度，降低通道数、高度和宽度
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

# 模块一使用7x7卷积层，接上批量规范化层，接上最大池化层
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 模块二到五为4个稠密块，每两个稠密块间加上一个过渡层减半高度、宽度、通道数
# num_channels为当前通道数
num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间加一个过渡层，使通道数减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2

net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10)
)

def test_densenet():
    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())


def test_densenet_s1():
    lr, num_epochs, batch_size = 0.1, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())
