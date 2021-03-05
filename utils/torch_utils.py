import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    # PE 层的前一层是embedding，维度是一般为 batch_size ，seq_len, embedding_dim
    # 一个句子 10个单词，每个单词 100维度表示，所以为 10 * 100
    # 参数 d_model 为 embedding的维度
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        # pe: max_len  * d_model
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #  position: max_len * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # div_term: d_model / 2
        pe[:, 0::2] = torch.sin(position * div_term)
        #  max_len * (d_model / 2)
        pe[:, 1::2] = torch.cos(position * div_term)
        #  max_len * (d_model / 2)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe: max_len * 1 * d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
