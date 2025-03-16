import torch 
from torch import nn
from torch.nn import functional as F
from utils.rnn.seqDataLoader import load_data_time_machine
from utils.gpu import try_gpu
from recurrentNeuralNetwork.rnn_train_ch8 import train_ch8, RNNModelScratch
from recurrentNeuralNetwork.rnn_sapi import RNNModel

# 双向循环神经网络
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

def test_bi_lstm_sapi():
    vocab_size, num_hiddens, num_layers, device = len(vocab), 256, 2, try_gpu()
    num_epochs, lr = 500, 1
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    model = RNNModel(lstm_layer, len(vocab))
    model = model.to(device)
    train_ch8(model, train_iter, vocab, lr, num_epochs, device)