import torch
import torch.nn as nn
from torch.nn.modules import TransformerDecoderLayer

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
src = torch.rand(10, 32, 512)
out = encoder_layer(src)
print(out.shape)

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = decoder_layer(tgt, memory)
print(out.shape)

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)
print(out.shape)