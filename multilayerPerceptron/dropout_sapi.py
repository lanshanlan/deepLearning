import torch
from torch import nn
from utils.loadData import load_data_fashion_mnist
from utils.train_ch3 import train_ch3, predict_ch3, evaluate_loss
from utils.loadData import synthetic_data, load_array
from utils.activatefunc import linreg, squared_loss, sgd
from utils.train_ch3 import Animator

# 暂退法pytorch简洁实现
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
num_epochs, lr, batch_size = 10, 0.5, 256
dropout1, dropout2 = 0.2, 0.5

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    # 在第一个全连接层后添加一个暂退层
    nn.Dropout(dropout1),
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    # 在第二个全连接层后添加一个暂退层
    nn.Dropout(dropout2),
    nn.Linear(num_hiddens2, num_outputs)
)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
loss = nn.CrossEntropyLoss(reduction='none')

def dropout_train_sapi():
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)