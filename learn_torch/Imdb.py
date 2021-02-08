# https://aadmirer.com/articles-72.html
# https://samaelchen.github.io/pytorch_lstm_sentiment/
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import torch.optim as optim
from torchtext.datasets import IMDB

# Field类定义了数据如何被处理，用TEXT来处理评论
TEXT = torchtext.data.Field()
# LABEL处理标签数据，其中LabelField是Field专门用来处理标签的子类
LABEL = torchtext.data.LabelField(dtype=torch.float)

# 划分训练集和测试集
train_data, test_data = IMDB.splits(TEXT, LABEL)

# 最大的词的数量
MAX_VOCAB_SIZE = 25000

# 建立词向量
TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

BATCH_SIZE = 32

# 定义device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据迭代器
train_iter, test_iter = torchtext.data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)


class IMDb_NET(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        # Embedding
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # LSTM
        self.lstm = nn.RNN(embedding_dim, hidden_dim, num_layers=2, dropout=0.2)
        # FC
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        embedding = self.embedding(x)
        output, hidden = self.lstm(embedding)
        return torch.sigmoid(self.fc(hidden[0].squeeze(0)))


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 80
HIDDEN_DIM = 128
OUTPUT_DIM = 1

net = IMDb_NET(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)


# Binary Cross Entropy
criterion = nn.BCELoss().to(device)
# adam优化器
optimizer = optim.Adam(net.parameters(), lr=0.01)


EPOCHS = 10

for epoch in range(EPOCHS):
    net.train()

    for i, data in enumerate(train_iter):
        text, label = data.text, data.label
        output = net(text).squeeze(1)
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('Epoch: {}, Step: {}, Loss: {}'.format(epoch, i, loss.item()))