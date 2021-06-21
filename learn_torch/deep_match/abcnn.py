# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:43:12 2020
@author: zhaog
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'abcnn'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.kernel_size = 2
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001
        self.max_seq_len = 32
        self.mp_hidden = 128
        self.linear_size = 20


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.embed = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.linear_size = config.linear_size
        self.num_layer = config.num_layer
        self.embeds_dim = config.embedding_dim
        self.conv = nn.ModuleList(
            [Wide_Conv(config.max_seq_len, config.embedding_dim) for _ in range(self.num_layer)])
        self.fc = nn.Sequential(
            nn.Linear(self.embeds_dim * (1 + self.num_layer) * 2, self.linear_size),
            nn.LayerNorm(self.linear_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.linear_size, config.out_features),
        )

    def forward(self, q1, q2):
        # batch_size,seq_length
        mask1, mask2 = q1.eq(0), q2.eq(0)
        res = [[], []]
        q1_encode = self.embed(q1)
        q2_encode = self.embed(q2)
        # eg: s1 => res[0]
        # (batch_size, seq_len, dim) => (batch_size, dim)
        # if num_layer == 0
        res[0].append(F.avg_pool1d(q1_encode.transpose(1, 2), kernel_size=q1_encode.size(1)).squeeze(-1))
        res[1].append(F.avg_pool1d(q2_encode.transpose(1, 2), kernel_size=q2_encode.size(1)).squeeze(-1))
        for i, conv in enumerate(self.conv):
            o1, o2 = conv(q1_encode, q2_encode, mask1, mask2)
            res[0].append(F.avg_pool1d(o1.transpose(1, 2), kernel_size=o1.size(1)).squeeze(-1))
            res[1].append(F.avg_pool1d(o2.transpose(1, 2), kernel_size=o2.size(1)).squeeze(-1))
            o1, o2 = attention_avg_pooling(o1, o2, mask1, mask2)
            q1_encode, q2_encode = o1 + q1_encode, o2 + q2_encode
        # batch_size * (dim*(1+num_layer)*2) => batch_size * linear_size
        x = torch.cat([torch.cat(res[0], 1), torch.cat(res[1], 1)], 1)
        sim = self.fc(x)
        return sim


class Wide_Conv(nn.Module):
    def __init__(self, seq_len, embeds_size):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size)))
        nn.init.xavier_normal_(self.W)
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()

    def forward(self, sent1, sent2, mask1, mask2):
        '''
        sent1, sent2: batch_size * seq_len * dim
        '''
        # sent1, sent2 = sent1.transpose(0, 1), sent2.transpose(0, 1)
        # => A: batch_size * seq_len * seq_len
        A = match_score(sent1, sent2, mask1, mask2)
        # attn_feature_map1: batch_size * seq_len * dim
        attn_feature_map1 = A.matmul(self.W)
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)
        # x1: batch_size * 2 *seq_len * dim
        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2


def match_score(s1, s2, mask1, mask2):
    '''
    s1, s2:  batch_size * seq_len  * dim
    mask1, mask2: batch_size * seq_len
    '''
    batch, seq_len, dim = s1.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float()
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1)
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1)
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)


def attention_avg_pooling(sent1, sent2, mask1, mask2):
    # A: batch_size * seq_len * seq_len
    A = match_score(sent1, sent2, mask1, mask2)
    weight1 = torch.sum(A, -1)
    weight2 = torch.sum(A.transpose(1, 2), -1)
    s1 = sent1 * weight1.unsqueeze(2)
    s2 = sent2 * weight2.unsqueeze(2)
    s1 = F.avg_pool1d(s1.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s2 = F.avg_pool1d(s2.transpose(1, 2), kernel_size=3, padding=1, stride=1)
    s1, s2 = s1.transpose(1, 2), s2.transpose(1, 2)
    return s1, s2


if __name__ == '__main__':
    num_embeddings = 5000
    batch_size = 128
    max_length = 32
    embedding_dim = 300
    out_features = 10
    epoch = 5
    config = Config(num_embeddings, embedding_dim, out_features)
    model = Model(config)
    print(model)
    print("---------------------")
    for param in model.named_parameters():
        print(param[0], param[1].shape)
    inputs1 = torch.randint(high=200, size=(batch_size, 30))
    m = nn.ZeroPad2d((0, 2, 0, 0))
    # m = nn.ZeroPad2d((left, right, top, bottom))
    inputs1 = m(inputs1)
    inputs2 = torch.randint(high=200, size=(batch_size, max_length))
    res = (model(inputs1, inputs2))
    assert res.shape == (batch_size, out_features)
