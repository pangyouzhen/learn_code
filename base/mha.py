# https://www.cnblogs.com/xiximayou/p/13343856.html
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from learn_torch.transformer_ import clones


def attention(query, key, value, mask=None, dropout=None):
    # query size : (batch,注意力头的个数,nums_seq, ??) 最后一个维度不清楚是啥
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # output: [30, 8, 10, 11]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # output: [30, 8, 10, 11]
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    # output: [30, 8, 10, 64]


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, ):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # 和 from torch.nn.modules.activation import MultiheadAttention 是一样的
        # query:`(L, N, E)`, key: :math:`(S, N, E)`, value: :math:`(S, N, E)` -> math:`(L, N, E)`
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # 这里是前三个Linear Networks的具体应用，
        # 例如query=(30,10, 512) -> Linear network -> (30, 10, 512)
        # -> view -> (30,10, 8, 64) -> transpose(1,2) -> (30, 8, 10, 64)
        # ，其他的key和value也是类似地，
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
          # 得到(30, 10, 512).


if __name__ == '__main__':
    query = torch.randn([30, 10, 512])
    key = torch.randn([30, 11, 512])
    value = torch.randn([30, 11, 512])
    # res, p_attn = attention(query, key, value)
    # print(res.shape, p_attn.shape)
    mha = MultiHeadAttention(8, 512)
    print(mha(query, key, value))
