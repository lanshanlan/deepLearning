import collections
import re
from utils.rnn.downloadFile import DATA_HUB, DATA_URL, download, download_extract
from utils.plotfunc import plot

DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                            '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def test_read_time_machine():
    lines = read_time_machine()
    print(f'# 文本总行数:{len(lines)}')
    print(lines[0])
    print(lines[10])

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型' + token)

def test_tokenize():
    lines = read_time_machine()
    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

def count_corpus(tokens):
    """统计词元出现的频率"""
    # 这里的tokens是一维列表或者二维列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    # print('tokens2', tokens)
    return collections.Counter(tokens)

class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return  self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    @property
    def unk(self): # 未知词元的索引为0
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
    
def test_vocab():
    lines = read_time_machine()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    print('list(vocab.token_to_idx.items())[:10]:\n ', list(vocab.token_to_idx.items())[:10])
    print('vocab.token_freq[:10]:\n ', vocab.token_freqs[:10])
    print('len(vocab.idx_to_token):\n ', len(vocab.idx_to_token))
    print('len(vocab.token_freqs):\n ', len(vocab.token_freqs))
    print('vocab.idx_to_token[:10]:\n ', vocab.idx_to_token[:10])

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    # tokens = tokenize(lines, 'char')
    tokens = tokenize(lines)
    # print('tokens:',tokens)
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，所以将所有文本行展平道一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def test_load_corpus_time_machine():
    corpus , vocab = load_corpus_time_machine()
    # print(len(corpus), len(vocab), vocab.token_freqs)
    # print(corpus[:50])
    freqs = [freq for token, freq in vocab.token_freqs]
    plot(freqs, xlabel='token: x', ylabel='frequency: n(x)', xscale='log', yscale='log')

# 测试不同的词元语法（一元语法，二元语法，三元语法）下，词元和频率的关系
def test_multi_tokens():
    lines = read_time_machine()
    tokens = tokenize(lines)
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    # 二元语法
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    # 三元语法
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print('bigram_vocab.token_freqs[:10]:', bigram_vocab.token_freqs[:10])
    print('trigram_vocab.token_freqs[:10]:', trigram_vocab.token_freqs[:10])
    freqs = [freq for token, freq in vocab.token_freqs]
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])


