import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

torch.manual_seed(1)


class ResBlock(nn.Module):
    # todo
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1, bias=False)
        pass

    def forward(self):
        pass
