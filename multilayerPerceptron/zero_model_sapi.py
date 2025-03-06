import torch
from torch import nn
from utils.loadData import load_data_fashion_mnist
from utils.train_ch3 import train_ch3, predict_ch3

# 多层感知机通过pytorch高级api简洁实现
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
train_iter, test_iter = load_data_fashion_mnist(batch_size)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)

def zero_model_sapi_train():
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
