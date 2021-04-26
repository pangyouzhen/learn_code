from collections import Counter
from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn import metrics
from torchtext.vocab import Vocab

train_df = pd.read_csv("/data/project/nlp_summary/data/THUCNews/data/train.txt", sep="\t", names=["sentence", "label"])
dev_df = pd.read_csv("/data/project/nlp_summary/data/THUCNews/data/dev.txt", sep="\t", names=["sentence", "label"])
# 数据探查
train_df["context_len"] = train_df["sentence"].apply(len)
print(train_df.describe())
train_df = train_df[(train_df['context_len'] <= 32)]
del train_df["context_len"]


def tokenizer(text: str) -> List:
    return [tok for tok in text]


train_df["sen_"] = train_df["sentence"].apply(tokenizer)
a = train_df["sen_"].tolist()
c = Counter([j for i in a for j in i])
vocab = Vocab(c)
print(vocab)
vectors_name = "sgns.sogounews.bigram-char"
vectors_path = "../../data/"

batch_size = 128
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 使用本地词向量
# torchtext.Vectors 会自动识别 headers
# vectors: Vectors = Vectors(name=vectors_name, cache=vectors_path)
# 获取词向量的维度
# vectors_dim: int = vectors.dim
# 获取分类的维度
# print("词向量的维度是", vectors_dim, "分类的维度是", num_class)
# SENTENCE.build_vocab(train, vectors=vectors)  # , max_size=30000)

class TextLstm(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, out_features):
        super(TextLstm, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, batch_first=True,
                            dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=out_features)

    def forward(self, sentence):
        # batch_size, seq_length
        x = self.embedding(sentence)
        # batch_size, seq_length, embedding_dim
        x, _ = self.lstm(x)
        # batch_size,seq_length,hidden_size
        out = x[:, -1, :]
        # batch_size,1,hidden_size
        pre_label = torch.softmax(self.linear(out), dim=-1)
        return pre_label


model = TextLstm(num_embeddings=300, embedding_dim=300,
                 hidden_size=128, out_features=10)
# print(SENTENCE.vocab.vectors.shape)
# model.embedding.weight.data.copy_(SENTENCE.vocab.vectors)
model.to(device)

lr = 0.001
crition = F.cross_entropy
# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # ,lr=0.000001)
model.train()
n_epoch = 20
for epoch in range(n_epoch):
    index = 0
    train_epoch_loss = 0
    train_acc = 0
    # Example object has no attribute sentence2，看前面 assert 那个
    for epoch2, batch in enumerate(train_iter):
        model.zero_grad()
        target = batch.label
        # target.shape == 128
        target = target.to(device)
        sentence = batch.sentence
        # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
        sentence = sentence.permute(1, 0)
        out = model(sentence)
        loss = crition(out, target)
        loss.backward()
        optimizer.step()
        index += 1
        if index % 50 == 0:
            # 每多少轮输出在训练集和验证集上的效果
            true = target.data.cpu()
            predic = torch.max(out.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predic)
            logger.info(f'epoch:{epoch} batch:{index} loss:{loss} train_acc:{train_acc}')
            # if dev_acc > best_acc:
            #     torch.save(model, f'{output_path}/{model_name}/model.pt')
            model.train()
    logger.info("---------------")

    for epoch2, batch in enumerate(dev_iter):
        target = batch.label
        # target.shape == 128
        target = target.to(device)
        sentence = batch.sentence
        # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
        sentence = sentence.permute(1, 0)
        out = model(sentence)
        loss = crition(out, target)
        index += 1
        if index % 2 == 0:
            # 每多少轮输出在训练集和验证集上的效果
            true = target.data.cpu()
            predic = torch.max(out.data, 1)[1].cpu()
            # predic = torch.max(out.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predic)
            # dev_acc = evaluate(model, dataloader_dev)
            logger.info(f'epoch:{epoch} batch:{index} loss:{loss} dev_acc:{train_acc}')
            # if dev_acc > best_acc:
            #     torch.save(model, f'{output_path}/{model_name}/model.pt')
    logger.info("---------------")
