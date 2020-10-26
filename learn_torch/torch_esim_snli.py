# 后续需要参考下这篇文章：https://zhuanlan.zhihu.com/p/64934558
import re

import torch
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors

from learn_torch.torch_esim_model import Esim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print("--------------------------------")
print(DEVICE)


def extract_text(s):
    # 移除括号
    s = re.sub('\\(', '', s)
    s = re.sub('\\)', '', s)
    # 使用一个空格替换两个以上连续空格
    s = re.sub('\\s{2,}', ' ', s)
    return s.strip().split()


LABEL = data.Field(sequential=False, use_vocab=True)
SENTENCE1 = data.Field(sequential=True, tokenize=extract_text, lower=True)
SENTENCE2 = data.Field(sequential=True, tokenize=extract_text, lower=True)

train = data.TabularDataset('/data/Downloads/data/dataset/snli_1.0/snli_1.0_train.txt', format='tsv', skip_header=True,
                            fields=[("gold_label", LABEL),
                                    ("sentence1_binary_parse", None), ("sentence2_binary_parse", None),
                                    ("sentence1_parse", None), ("sentence2_parse", None),
                                    ('sentence1', SENTENCE1), ('sentence2', SENTENCE2),
                                    ("captionID", None), ("pairID", None), ("label1", None), ("label2", None),
                                    ("label3", None), ("label4", None), ("label5", None)])

# 增加读取文件类型判断
assert list(train[5].__dict__.keys()) == ['gold_label', 'sentence1', 'sentence2']
print(train[5])
print(train[5].__dict__.keys())
print(train.examples[1].sentence2)

# 使用本地词向量
# torchtext.Vectors 会自动识别 headers
vectors = Vectors(name="glove.6B.100d.txt", cache="/data/project/learn_allennlp/data/.vector_cache/")

# 获取词向量的维度
vectors_dim = vectors.dim
# 获取分类的维度
print(set([i.gold_label for i in train.examples]))
num_class = len(set([i.gold_label for i in train.examples]))
# 这里怎样去除不想要的类别
print("词向量的维度是", vectors_dim, "分类的维度是", num_class)

LABEL.build_vocab(train)
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
sentence2_vocab = len(SENTENCE2.vocab)

# print(SENTENCE1.vocab.vectors.shape) learn_torch.Size([1443, 300])
# 查看一个batch
# batch = next(iter(train_iter))
# print(batch.fields)
# print("batch text: ", batch.sentence1.shape) -> [64, 128] (seq_num_a,batch_size)
# print("batch text: ", batch.sentence2.shape) -> [63, 128]
# print("batch label: ", batch.label.shape) -> [128]

model = Esim(sentence1_vocab=sentence1_vocab, sentence2_vocab=sentence2_vocab, embedding_dim=100, hidden_size=20,
             num_class=num_class)
print(SENTENCE1.vocab.vectors.shape)
model.embedding1.weight.data.copy_(SENTENCE1.vocab.vectors)
model.embedding2.weight.data.copy_(SENTENCE2.vocab.vectors)
model.to(DEVICE)

crition = F.cross_entropy
# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # ,lr=0.000001)
# 学习率过大，会导致震荡，甚至无法收敛
# 学习率过小，训练缓慢
n_epoch = 10

best_val_acc = 0

for epoch in range(n_epoch):
    epoch_loss = 0
    train_acc = 0
    # Example object has no attribute sentence2，看前面 assert 那个
    for epoch2, batch in enumerate(train_iter):
        target = batch.gold_label
        # target.shape == 128
        target = target.to(DEVICE)
        optimizer.zero_grad()
        sentence1 = batch.sentence1
        # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
        sentence1 = sentence1.permute(1, 0)
        sentence2 = batch.sentence2
        sentence2 = sentence2.permute(1, 0)

        out = model(sentence1, sentence2)
        # print("---------------------------")
        # target 默认构建的label是从 1 开始的
        target = target - 1
        loss = crition(out, target)
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.data
        y_pre = torch.argmax(out, dim=-1)
        train_acc += (torch.argmax(out, dim=-1) == target).sum().item()
        # print("train_acc", train_acc)
    print("epoch_loss is", epoch_loss / len(train), "acc is", train_acc / len(train))
#  如果cuda 下报错，最好先看下 cpu是否运行正确
