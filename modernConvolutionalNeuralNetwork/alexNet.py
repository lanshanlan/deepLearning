import torch
from torch import nn
from utils.loadData import load_data_fashion_mnist
from utils.train_ch6 import train_ch6
from utils.print_layer import print_layer

batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

# AlexNet卷积神经网络
# ImageNet挑战赛在2012年第一名的模型
net = nn.Sequential(
    # 使用一个11x11的更大窗口捕捉对象,因为ImageNet图像的高和宽（224x224）是MNIST图像(28x28)的10倍以上
    # 同时，步幅为4，以减少输出的高度和宽度
    # 另外，输出通道数远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入和输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 使用连续3个卷积层和较小的卷积窗口
    # 除了最后的卷积层，输出通道数进一步增加
    # 在前两个卷积层后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍，使用暂退层来缓解过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。因为我们测试的是Fashion-MNIST，所以类别数是10，而非AlexNet论文中的1000
    nn.Linear(4096, 10)
)

def print_alex_net_layer():
    X_shape = (1, 1, 224, 224)
    print_layer(net, X_shape)

def test_AlexNet():
    lr, num_epochs = 0.01, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr)