import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer

r = torch.randn(10,32,512)
trm = TransformerEncoderLayer(512,8,batch_first=True)
print(trm(r).shape)