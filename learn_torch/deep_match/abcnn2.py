import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

torch.manual_seed(1)


class WideConv(nn.Module):
    def __init__(self, embedding_dim, a, b):
        super(WideConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=embedding_dim, kernel_size=(kernel, embedding_dim))

    def forward(self, sentence):
        pass
        # batch_size, seq_len, embedding_dim
