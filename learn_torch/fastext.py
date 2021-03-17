import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# [fasttext](https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/master/models)

#  TODO
class FastText(nn.Module):
    def __init__(self, num_mebedding, embedding_dim, droput, n_gram, hidden_size, num_class):
        super(FastText, self).__init__()
        self.num_mebedding = num_mebedding
        self.embedding_dim = embedding_dim
        self.dropoout = droput
        self.n_gram = n_gram
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.num_mebedding, self.embedding_dim)
        self.embedding2 = nn.Embedding(self.n_gram, self.embedding_dim)
        self.embedding3 = nn.Embedding(self.n_gram, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropoout)
        self.fc1 = nn.Linear(embedding_dim * 3, self.hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self):
        pass
