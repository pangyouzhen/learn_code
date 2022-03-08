import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
import torch.nn.functional as F


# todo
class TrmEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MA(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src):
        src2 = self.self_attn(src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# 多头注意力机制
class MA(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.qkv_para = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.out_proj_weight = nn.Parameter(torch.empty(embed_dim, embed_dim))

    def forward(self, src: torch.Tensor):
        # self-attention
        q, k, v = F.linear(src, self.qkv_para, self.bias).chunk(3, dim=-1)
        tgt_len, bsz, embed_dim = src.size()
        src_len = k.size(1)
        q = q.contiguous().view(tgt_len, bsz * self.nhead, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.nhead, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.nhead, self.head_dim).transpose(0, 1)
        attn_output_weights = q @ k.transpose(1, 2)
        attn_output_weights = F.softmax(
            attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=0.1)
        attn_output = attn_output_weights @ v
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj_weight)
        attn_output_weights = attn_output_weights.view(bsz, self.nhead, tgt_len, src_len)
        return attn_output, attn_output_weights


if __name__ == '__main__':
    trm_encoder = TrmEncoder(d_model=512, nhead=8)
    src = torch.rand(10, 32, 512)
    print(trm_encoder(src))
