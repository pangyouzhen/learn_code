# https://blog.csdn.net/baidu_15113429/article/details/104876726
# -*- coding: utf-8 -*-
# @Time : 2020/2/25 11:19
# @Author : liusen
from torch import nn
import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenizer(text):
    return [tok for tok in text]


LABEL = data.Field(sequential=False, use_vocab=False)
SENTENCE1 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
SENTENCE2 = data.Field(sequential=True, tokenize=tokenizer, lower=True)

train = data.TabularDataset('../full_data/ants/ants_torchtext_train.csv', format='csv', skip_header=True,
                            fields=[('sentence1', SENTENCE1), ('sentence2', SENTENCE2), ('label', LABEL)])
# 增加读取文件类型判断
assert list(train[5].__dict__.keys()) == ['sentence1', 'sentence2', 'label']

# 使用本地词向量
# torchtext.Vectors 会自动识别 headers
vectors = Vectors(name="sgns.sogounews.bigram-char", cache="./data/")
SENTENCE1.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE1.vocab.vectors.unk_init = init.xavier_uniform

SENTENCE2.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE2.vocab.vectors.unk_init = init.xavier_uniform

batch_size = 128

train_iter = data.BucketIterator(train, batch_size=batch_size,
                                 shuffle=True, device=DEVICE)
sentence1_vocab = len(SENTENCE1.vocab)


class ESIM(nn.Module):
    def __init__(self):
        super(ESIM, self).__init__()
        self.dropout = 0.5
        self.hidden_size = 128
        self.embeds_dim = 300
        self.embeds = nn.Embedding(len(SENTENCE1.vocab), self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(self.dropout),
            nn.Linear(2, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(self.dropout),
            nn.Linear(2, 2),
            nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        # mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        # mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        # weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        weight1 = F.softmax(attention, dim=-1)
        x1_align = torch.matmul(weight1, x2)
        # weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, *input):
        # batch_size * seq_len
        sent1, sent2 = input[0], input[1]
        mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        # x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        # x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2).transpose(0, 1)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2).transpose(0, 1)

        # batch_size * seq_len * dim =>      batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity


def val_model(val_iter, net3):
    confusion_matrix = torch.zeros(2, 2)
    for labels, batch in enumerate(val_iter):
        predicted = net3(batch.query1, batch.query2)
        prediction = torch.max(predicted, 1)[1].data.numpy()
        label = batch.label
        for t, p in zip(prediction, label):
            confusion_matrix[t, p] += 1
    a_p = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]
    print("被所有预测为负的样本中实际为负样本的概率", a_p)
    b_p = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
    print("被所有预测为正的样本中实际为正样本的概率", b_p)
    a_r = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]
    print("实际的负样本中预测为负样本的概率，召回率", a_r)
    b_r = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
    print("实际的正样本中预测为正样本的概率，召回率", b_r)
    f1 = 2 * a_p * a_r / (a_p + a_r)
    f2 = 2 * b_p * b_r / (b_p + b_r)
    return f1, f2


def main():
    model = ESIM().to(DEVICE)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    crition = F.cross_entropy
    min_f = 0
    for epoch in range(30):
        epoch_loss = 0
        for epoch, batch in enumerate(train_iter):
            optimizer.zero_grad()
            sentence1 = batch.sentence1
            # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
            sentence1 = sentence1.permute(1, 0)
            sentence2 = batch.sentence2
            sentence2 = sentence2.permute(1, 0)
            label = batch.label
            label.to(DEVICE)

            predicted = model(sentence1, sentence2)
            loss = crition(predicted, label)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.data
        # 计算每一个epoch的loss
        # 计算验证集的准确度来确认是否存储这个model
        print("epoch_loss", epoch_loss)
        # f1, f2 = val_model(val_iter, model)
        # if (f1 + f2) / 2 > min_f:
        #     min_f = (f1 + f2) / 2
        #     print("save model")
        #     learn_torch.save(model.state_dict(), '../data/esim_match_data/esim_params_30.pkl')


if __name__ == '__main__':
    main()
