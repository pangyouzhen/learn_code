import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# [fasttext](https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/master/models)

# todo
# fasttext 的输入不一样
class Config(object):
    """配置参数"""

    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'fastext'
        self.num_mebedding = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.dropout = 0.5  # 随机失活
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.lr = 1e-3  # 学习率
        self.hidden_size = 256  # 隐藏层大小
        self.n_gram_vocab = 250499  # ngram 词表大小


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_mebedding = config.num_mebedding
        self.embedding_dim = config.embedding_dim
        self.dropoout = config.dropout
        self.n_gram = config.n_gram_vocab
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(self.num_mebedding, self.embedding_dim)
        self.embedding_ngram2 = nn.Embedding(self.n_gram, self.embedding_dim)
        self.embedding_ngram3 = nn.Embedding(self.n_gram, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropoout)
        self.fc1 = nn.Linear(config.embedding_dim * 3, self.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.out_features)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        # fasttext 核心将 word,bigram,trigram 合并起来
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
