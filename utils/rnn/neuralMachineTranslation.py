import torch
import hashlib
import os
import hashlib
import tarfile
import zipfile
import requests
from utils.rnn.downloadFile import DATA_HUB, DATA_URL, download, download_all, download_extract
from utils.plotfunc import set_figsize
from utils.rnn.textPreprocessing import Vocab
from matplotlib import pyplot as plt
from utils.loadData import load_array

# 神经机器翻译数据：‘英语-法语’翻译数据集
DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                            '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """载入‘英语-法语’翻译数据集"""
    data_iter = download_extract('fra-eng')
    with open(os.path.join(data_iter, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()
    
def test_read_data_nmt():
    raw_text = read_data_nmt()
    print(raw_text[:75])

def preprocess_nmt(text):
    """预处理‘英语-法语’数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '
    # 用空格替换不间断空格
    # 用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1])
           else char for i, char in enumerate(text)]
    return  ''.join(out)

def test_preprocess_nmt():
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    print(text[:80])

def tokenize_nmt(text, num_examples=None):
    """词元化‘英语-法语’数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def test_tokenize_nmt():
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    source, target = tokenize_nmt(text)
    print(source[:6], target[:6])

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    set_figsize()
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]]
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)

def test_show_list_len_pair_hist():
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    source, target = tokenize_nmt(text)
    print(source[:6], target[:6])
    show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                            'count', source, target)
    
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps] # 截断
    return line + [padding_token] * (num_steps - len(line)) # 填充

raw_text = read_data_nmt()
text = preprocess_nmt(raw_text)
source, target = tokenize_nmt(text)
src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])

def test_truncate_pad():
    print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']
    ) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def test_load_data_nmt():
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break