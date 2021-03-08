from typing import List
from sklearn import metrics

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
from loguru import logger

from utils.DataFrameDataSet import DataFrameDataset
from learn_torch.torch_esim_model import Esim

logger.add("./log/ll.log")

train_df = pd.read_csv("../full_data/ants/ants_torchtext_train.csv", sep=",", encoding="utf-8")
train_df["sentence1_len"] = train_df["sentence1"].apply(len)
train_df["sentence2_len"] = train_df["sentence2"].apply(len)
print(train_df["sentence1_len"], train_df["sentence2_len"])
print(train_df.describe())
train_df = train_df[(train_df['sentence1_len'] <= 15) & (train_df['sentence2_len'] <= 15)]
train_df = train_df[["sentence1", "sentence2", "label"]]
print(train_df.shape)
vectors_name = "sgns.sogounews.bigram-char"
vectors_path = "../data/"


def tokenizer(text: str) -> List:
    return [tok for tok in text]


batch_size = 128
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL: data.Field = data.Field(sequential=False, use_vocab=False)
SENTENCE: data.Field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
train: DataFrameDataset = DataFrameDataset(train_df,
                                           fields={'sentence1': SENTENCE, 'sentence2': SENTENCE, "label": LABEL})
# 使用本地词向量
# torchtext.Vectors 会自动识别 headers
vectors: Vectors = Vectors(name=vectors_name, cache=vectors_path)
# 获取词向量的维度
vectors_dim: int = vectors.dim
# 获取分类的维度
num_class = len(set([i.label for i in train.examples]))
print("词向量的维度是", vectors_dim, "分类的维度是", num_class)
SENTENCE.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 这里SENTENCE 根据vectors 构建了自己的词向量 ->> SENTENCE.vocab.vectors
# 如果原始词向量中没有这个词-> 构建成一个0 tensor
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE.vocab.vectors.unk_init = init.xavier_uniform
train_iter: data.BucketIterator = data.BucketIterator(train, batch_size=batch_size,
                                                      shuffle=True, device=device)

model = Esim(sentence_vocab=len(SENTENCE.vocab), embedding_dim=SENTENCE.vocab.vectors.shape[-1], hidden_size=20,
             num_class=num_class)
print(SENTENCE.vocab.vectors.shape)
model.embedding.weight.data.copy_(SENTENCE.vocab.vectors)
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
        sentence1 = batch.sentence1
        # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
        sentence1 = sentence1.permute(1, 0)
        sentence2 = batch.sentence2
        sentence2 = sentence2.permute(1, 0)

        out = model(sentence1, sentence2)
        loss = crition(out, target)
        loss.backward()
        optimizer.step()
        index += 1
        if index % 50 == 0:
            # 每多少轮输出在训练集和验证集上的效果
            true = target.data.cpu()
            predic = torch.max(out.data, 1)[1].cpu()
            train_acc = metrics.accuracy_score(true, predic)
            # dev_acc = evaluate(model, dataloader_dev)
            logger.info(f'epoch:{epoch} batch:{index} loss:{loss} train_acc:{train_acc}')
            # if dev_acc > best_acc:
            #     torch.save(model, f'{output_path}/{model_name}/model.pt')
            model.train()
    logger.info("---------------")
        # train_epoch_loss = train_epoch_loss + loss.data
        # print((torch.argmax(out, dim=-1) == target).sum())
        # train_acc += (torch.argmax(out, dim=-1) == target).sum().item()
        # print("train_loss/epoch", train_epoch_loss / len(train), epoch)
        # print("train_acc/epoch", train_acc / len(train), epoch)
        # scheduler.step()
        # print("epoch is {} train_epoch_loss is {} train_acc is {}".format(epoch, train_epoch_loss,
