import torch
import torch.nn as nn
import torch.nn.functional as F


class RE2(nn.Module):
    def __init__(self, embedding_dim, dropout, num_embeddings, l):
        super(RE2, self).__init__()
        self.embeding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.l = l

    def forward(self, x):
        #  batch_size,seq_length
        x = self.embedding(x)
        #  batch_size, seq_length,embeding_dim


class Block(nn.Module):
    def __init__(self, embedding_dim, hidden_size, c, d):
        super(Block, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.c = c
        self.d = d
        #  TODO 这里的 是要结合前面的，所以只定义embedding_dim 相当于只能定义第一层
        self.encode = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size)

    def forward(self, x):
        # batch_size,seq_length,embedding
        pass
