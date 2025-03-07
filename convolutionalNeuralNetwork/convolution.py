import torch
from torch import nn

def corr2d(X, K):
    """二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

class Conv2D(nn.Module):
    """手写的二维卷积层"""
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def test_conv2d():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    lr = 3e-2
    X = X.reshape((1,1,6,8))
    Y = Y.reshape((1,1,6,7))
    net = nn.Conv2d(1,1,kernel_size=(1,2), bias=True)
    for i in range(10):
        Y_hat = net(X)
        l = (Y_hat - Y) ** 2
        net.zero_grad()
        l.sum().backward()
        net.weight.data[:] -= lr * net.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch{i + 1}, loss {l.sum():.3f}')
    print(f'weight:{net.weight.data}')

# 卷积计算函数
# 此函数初始化卷积层权重，并对输入和输出扩大和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1,1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 返回值省略前两个维度：批量大小和通道数
    return Y.reshape(Y.shape[2:])

def test_comp_conv2d():
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=3, stride=2)
    X = torch.rand(size=(8, 8))
    Y = comp_conv2d(conv2d, X)
    print(Y)

def corr2d_multi_in(X, K):
    # 多输入通道互相关运算
    # 先遍历X和K的第0个维度（通道维度），再把他们加到一起
    return sum(corr2d(x, k) for x, k in zip(X, K))

def test_corr2d_multi_in():
    X = torch.arange(18).reshape(2,3,3)
    K = torch.arange(8).reshape(2,2,2)
    Y = corr2d_multi_in(X,K)
    print(Y.shape, Y)

def corr2d_multi_in_out(X, K):
    # 多通道卷积核
    # 迭代K的第0个维度，每次都对X执行互相关运算
    # 最后将所有结果叠加到一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

def test_corr2d_multi_in_out():
    X = torch.arange(18).reshape(2,3,3)
    K = torch.arange(8).reshape(2,2,2)
    K = torch.stack((K, K+1, K+2), 0)
    Y = corr2d_multi_in_out(X, K)
    print(Y.shape, Y)
    