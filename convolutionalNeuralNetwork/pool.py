import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    """手写池化层"""
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

def test_pool2d():
    X = torch.arange(9, dtype=torch.float32).reshape(3,3)
    pool_size = (2, 2)
    Y = pool2d(X, pool_size, mode='avg')
    print(Y)

def test_Pool2d():
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    X1 = torch.cat((X, X + 1), 1)
    maxPool2d = nn.MaxPool2d(3, padding=1, stride=2)
    avgPool2d = nn.AvgPool2d(3, padding=1, stride=2)
    print(maxPool2d(X))
    print(maxPool2d(X1))
    print(avgPool2d(X))
