import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from learn_torch.other.PositionalEncoding import PositionalEncoding
from copy import deepcopy

torch.manual_seed(1)


class QANet(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, a, b):
        super(QANet, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.a = a
        self.b = b

    def forward(self):
        pass


class QaNetBlock(nn.Module):
    def __init__(self, embedding_dim, d_model=256, normalized_shape=5):
        super().__init__()
        self.pe = PositionalEncoding(d_model=d_model)
        self.conv = nn.Conv2d(in_channels=1, out_channels=embedding_dim, )
        self.layernorm = nn.LayerNorm(normalized_shape)
        # self.conv_layer = nn.ModuleList(
        #     [self.layernorm(x) ]
        # )

    def forward(self, x):
        # batch_size, seq_len, embedding_dim
        x = self.pe(x)
        y1 = x
        for i in range(10):
            x = self.layernorm(x)
            x = self.conv(x)
            x += y1
        y2 = x
        x = self.layernorm(x)
        pass


class LayerConv(nn.Module):
    def __init__(self, embedding_dim, out_channels):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=5)
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(2, embedding_dim))

    def forward(self, x):
        # batch_size, seq_len, embedding_dim
        y = deepcopy(x)
        x = self.layernorm(x)
        #  batch_size, seq_len, embedding_dim
        x = x.unsqueeze(1)
        #  batch_size,1, seq_len, embedding_dim
        x = self.conv(x)
        # batch_size,out_channels, seq_len - kernel_size+ 1, 1
        x += y
        return x
