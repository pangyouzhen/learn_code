import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

torch.manual_seed(1)


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'match_pyramid'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.kernel_size = 2
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim)
        # todo
        self.cnn = nn.Conv2d(in_channels=1, out_channels=config.seq_length,
                             kernel_size=(config.kernel_size, config.embedding_dim))

    def forward(self, a, b):
        # batch_size, seq_length
        a = self.embedding(a)
        b = self.embedding(b)
        # batch_size,seq_length,embedding
        match_matrix = torch.matmul(a, b.transpose(2, 1))
        # batch_size,seq_length_a,seq_length_b
        match_matrix = match_matrix.unsqueeze(1)
        # batch_size,1,seq_length_b,seq_length_a - kernel_size + 1


    def conv_and_pool(self, x, conv):
        # x: batch_size,1,seq_length,embedding_dim
        # conv: 1, embedding_dim,(kernel,embedding_dim)
        # result: batch_size, embedding_dim,
        x = F.relu(conv(x)).squeeze(-1)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        return x
