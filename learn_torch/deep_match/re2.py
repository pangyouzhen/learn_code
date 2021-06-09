import torch
import torch.nn as nn
import torch.nn.functional as F


class RE2(nn.Module):
    def __init__(self, embedding_dim, dropout, num_embeddings, l):
        super(RE2, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encode1 = nn.LSTM(input_size=embedding_dim, hidden_size=128, batch_first=True, num_layers=2,
                               bidirectional=True)
        self.encode2 = nn.LSTM(input_size=embedding_dim, hidden_size=128, batch_first=True, num_layers=2,
                               bidirectional=True)
        self.encode = nn.LSTM(input_size=embedding_dim, hidden_size=128, batch_first=True, num_layers=2,
                              bidirectional=True)
        self.dropout = dropout

    def forward(self, x, y):
        pass

    def encode1(self, x):
        #  batch_size,seq_length
        x = self.embedding(x)
        #  batch_size, seq_length,embeding_dim
        encode1 = self.encode1(x)
        #  batch_size,seq_length,hidden_size * 2
        return encode1

    def encode2(self):
        pass

    def encode(self, x, y):
        # encoder
        res_x, res_y = x, y
        x, y = self.encode(x), self.encode(y)
        # 残差连接
        x += res_x
        y += res_y
        # 对齐层


        # Fusion Layer（融合层）

    def alignment(self, x, y):
        # x,y: batch_size, seq_length, embedding_dim
        pass
