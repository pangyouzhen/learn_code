# https://blog.csdn.net/qq_44722174/article/details/104640018
import json

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors
from torch.utils.tensorboard import SummaryWriter

from learn_torch.torch_esim_model import Esim
from learn_torch.DataFrameDataSet import DataFrameDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("--------------------------------")
print(DEVICE)


# 预处理，生成torchtext的格式
def process_dataset(file):
    def split_multi_columns(x):
        js = json.loads(x["json"])
        return js["sentence1"], js["sentence2"], js["label"]

    df = pd.read_csv(file, sep="\t", encoding="utf-8", names=["json"])
    # 别忘了result_type 参数
    df[['sentence1', 'sentence2', 'label']] = df.apply(lambda x: split_multi_columns(x), axis=1, result_type="expand")
    df = df[['sentence1', 'sentence2', 'label']]
    df.to_csv("./full_data/ants/ants_torchtext_train.csv", index=False, encoding="utf-8")


writer = SummaryWriter()


def tokenizer(text):
    return [tok for tok in text]


# quotechar 参数，比如句子中只有一个",数据错乱的情况下
df = pd.read_csv("../full_data/ants/ants_torchtext_train.csv", sep=",", encoding="utf-8")
# print(df[:5])
# print(df["label"].value_counts())
assert df["label"].unique().shape == 2
LABEL = data.Field(sequential=False, use_vocab=False)
SENTENCE1 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
SENTENCE2 = data.Field(sequential=True, tokenize=tokenizer, lower=True)

# train = data.TabularDataset('../full_data/ants/ants_torchtext_train.csv', format='csv', skip_header=True,
#                             fields=[('sentence1', SENTENCE1), ('sentence2', SENTENCE2), ('label', LABEL)])
train = DataFrameDataset(df, fields={'sentence1': SENTENCE1, 'sentence2': SENTENCE2, "label": LABEL})
# 增加读取文件类型判断
assert list(train[5].__dict__.keys()) == ['sentence1', 'sentence2', 'label']

# 使用本地词向量
# torchtext.Vectors 会自动识别 headers
vectors = Vectors(name="sgns.sogounews.bigram-char", cache="../data/")
# 获取词向量的维度
vectors_dim = vectors.dim
# 获取分类的维度
num_class = len(set([i.label for i in train.examples]))
print("词向量的维度是", vectors_dim, "分类的维度是", num_class)

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


model = Esim(sentence1_vocab=sentence1_vocab, sentence2_vocab=sentence2_vocab, embedding_dim=vectors_dim,
             hidden_size=20, num_class=num_class)
print(SENTENCE1.vocab.vectors.shape)
model.embedding1.weight.data.copy_(SENTENCE1.vocab.vectors)
model.embedding2.weight.data.copy_(SENTENCE2.vocab.vectors)
model.to(DEVICE)

crition = F.cross_entropy
# 训练
optimizer = torch.optim.Adam(model.parameters())  # ,lr=0.000001)

n_epoch = 20

best_val_acc = 0


def training(model, n_epoch, train_iter):
    for epoch in range(n_epoch):
        epoch_loss = 0
        train_acc = 0
        # Example object has no attribute sentence2，看前面 assert 那个
        for epoch2, batch in enumerate(train_iter):
            target = batch.label
            # target.shape == 128
            target = target.to(DEVICE)
            optimizer.zero_grad()
            sentence1 = batch.sentence1
            # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
            sentence1 = sentence1.permute(1, 0)
            sentence2 = batch.sentence2
            sentence2 = sentence2.permute(1, 0)

            out = model(sentence1, sentence2)
            loss = crition(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.data
            train_acc += (torch.argmax(out, dim=-1) == target).sum().item()
        writer.add_scalar("Loss/epoch", epoch_loss, epoch)
        print("epoch_loss is", epoch_loss, "acc is", train_acc / len(train))


if __name__ == '__main__':
    training(model, 20, train_iter)
