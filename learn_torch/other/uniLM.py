import torch
import torch.nn as nn
import torch.nn.functional as F


def attn(a, b):
    #  batch_size, seq_length, embedding
    # 交互矩阵计算
    attn = torch.matmul(a, b.transpose(2, 1))
    #  batch_size,seq_length_a,seq_length_b
    attn_a = F.softmax(attn, dim=1)
    #  batch_size,seq_length_a,seq_length_b
    attn_b = F.softmax(attn, dim=2)
    #  batch_size,seq_length_a,seq_length_b
    feature_b = torch.matmul(attn_a.transpose(1, 2), a)
    # batch_size, seq_length_b, embedding
    feature_a = torch.matmul(attn_b, b)
    return feature_a, feature_b
