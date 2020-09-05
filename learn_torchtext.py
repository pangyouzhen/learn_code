import pandas as pd
from torch import optim

# data = pd.read_csv('./data/train.tsv', sep='\t')
#
# print(data[:5])
# print(data.columns)
# Index(['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'], dtype='object')

import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
import torch.nn as nn
import torch.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spacy_en = spacy.load('en')


def tokenizer(text):  # create a tokenizer function
    """
    定义分词操作
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


"""
field在默认的情况下都期望一个输入是一组单词的序列，并且将单词映射成整数。
这个映射被称为vocab。如果一个field已经被数字化了并且不需要被序列化，
可以将参数设置为use_vocab=False以及sequential=False。
"""
LABEL = data.Field(sequential=False, use_vocab=False)

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)

"""
我们不需要 'PhraseId' 和 'SentenceId'这两列, 所以我们给他们的field传递 None
如果你的数据有列名，如我们这里的'Phrase','Sentiment',...
设置skip_header=True,不然它会把列名也当一个数据处理
"""
train, val = data.TabularDataset.splits(
    path='.', train='train.csv', validation='val.csv', format='csv', skip_header=True,
    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)])

test = data.TabularDataset('test.tsv', format='tsv', skip_header=True,
                           fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)])
TEXT.build_vocab(train, vectors='glove.6B.100d')  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
TEXT.vocab.vectors.unk_init = init.xavier_uniform
#  第1510 个词
print(TEXT.vocab.itos[1510])
#  bore对应词的index
print(TEXT.vocab.stoi['bore'])
# 词向量矩阵: TEXT.vocab.vectors
print(TEXT.vocab.vectors.shape)
word_vec = TEXT.vocab.vectors[TEXT.vocab.stoi['bore']]
print(word_vec.shape)
print(word_vec)

# 6. 构造迭代器

train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.Phrase),
                                 shuffle=True, device=DEVICE)

val_iter = data.BucketIterator(val, batch_size=128, sort_key=lambda x: len(x.Phrase),
                               shuffle=True, device=DEVICE)

# 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
test_iter = data.Iterator(dataset=test, batch_size=128, train=False,
                          sort=False, device=DEVICE)

"""
由于目的是学习torchtext的使用，所以只定义了一个简单模型
"""
len_vocab = len(TEXT.vocab)


class Enet(nn.Module):
    def __init__(self):
        super(Enet, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 100)
        self.lstm = nn.LSTM(100, 128, 3, batch_first=True)  # ,bidirectional=True)
        self.linear = nn.Linear(128, 5)

    def forward(self, x):
        batch_size, seq_num = x.shape
        vec = self.embedding(x)
        out, (hn, cn) = self.lstm(vec)
        out = self.linear(out[:, -1, :])
        out = F.softmax(out, -1)
        return out


model = Enet()
"""
将前面生成的词向量矩阵拷贝到模型的embedding层
这样就自动的可以将输入的word index转为词向量
"""
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.to(DEVICE)

# 训练
optimizer = optim.Adam(model.parameters())  # ,lr=0.000001)

n_epoch = 20

best_val_acc = 0

for epoch in range(n_epoch):

    for batch_idx, batch in enumerate(train_iter):
        data = batch.Phrase
        target = batch.Sentiment
        target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        data = data.permute(1, 0)
        optimizer.zero_grad()

        out = model(data)
        loss = -target * torch.log(out) - (1 - target) * torch.log(1 - out)
        loss = loss.sum(-1).mean()

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 200 == 0:
            _, y_pre = torch.max(out, -1)
            acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
            print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f'
                  % (epoch, batch_idx, loss, acc))

    val_accs = []
    for batch_idx, batch in enumerate(val_iter):
        data = batch.Phrase
        target = batch.Sentiment
        target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        data = data.permute(1, 0)
        out = model(data)

        _, y_pre = torch.max(out, -1)
        acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
        val_accs.append(acc)

    acc = np.array(val_accs).mean()
    if acc > best_val_acc:
        print('val acc : %.4f > %.4f saving model' % (acc, best_val_acc))
        torch.save(model.state_dict(), 'params.pkl')
        best_val_acc = acc
    print('val acc: %.4f' % (acc))
