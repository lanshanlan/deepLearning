import torch
from torch import nn
from utils.loadData import load_data_fashion_mnist
from utils.train_ch6 import train_ch6
from utils.print_layer import print_layer
from utils.gpu import try_gpu
from torch.nn import functional as F

# ResNet残差网络
# ImageNet挑战赛2015年第一名的模型
# 设计了残差块（residual block）

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
def print_Residual_shape():
    blk = Residual(3,3)
    X = torch.rand(4,3,6,6)
    print(blk(X).shape)
    blk2 = Residual(3,6,use_1x1conv=True, strides=2)
    print(blk2(X).shape)

# 模块一是7x7卷积层，接上批量规范化层，接上步幅为2的3x3最大汇聚层
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

# 残差块
def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 模块二到五都是残差块
# 注意第二个模块通道数不改，后面三个模块通道数翻倍而高宽减半，
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, 10))

def print_resnet_layer():
    X_shape = (1, 1, 96, 96)
    print_layer(net, X_shape)
def test_resnet():
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

def test_resnet_s1():
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())