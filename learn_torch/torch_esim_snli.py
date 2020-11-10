import re

import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
from torchtext import data
from torchtext.vocab import Vectors

from learn_torch.DataFrameDataSet import DataFrameDataset
from learn_torch.torch_esim_model import Esim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("--------------------------------")
print(DEVICE)


def extract_text(s):
    # 移除括号
    s = re.sub('\\(', '', s)
    s = re.sub('\\)', '', s)
    # 使用一个空格替换两个以上连续空格
    s = re.sub('\\s{2,}', ' ', s)
    return s.strip()


spacy_en = spacy.load('en_core_web_md')

label_set = {

    'entailment': 0, 'contradiction': 1, 'neutral': 2}


def tokenizer(text):  # create a tokenizer function
    """
    定义分词操作
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


LABEL = data.Field(sequential=False, use_vocab=False)
SENTENCE1 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
SENTENCE2 = data.Field(sequential=True, tokenize=tokenizer, lower=True)

data_dir = "/data/Downloads/data/dataset/snli_1.0/snli_1.0_train.txt"
df = pd.read_csv(data_dir, sep="\t")
df1 = df[["gold_label", "sentence1_parse", "sentence2_parse"]]
df1["sentence1_parse"] = df1["sentence1_parse"].apply(extract_text)
df1["sentence2_parse"] = df1["sentence2_parse"].apply(extract_text)
df1 = df1[df1["gold_label"].apply(lambda x: x in label_set.keys())]
df1["gold_label"] = df1["gold_label"].map(label_set)
assert df1["gold_label"].unique().shape == (3,)
train = DataFrameDataset(df1, fields={'sentence1_parse': SENTENCE1, 'sentence2_parse': SENTENCE2, "gold_label": LABEL})
vectors = Vectors(name="glove.6B.100d.txt", cache="/data/project/learn_allennlp/data/.vector_cache/")
# 获取词向量的维度
vectors_dim = vectors.dim
# 获取分类的维度
num_class = len(set([i.gold_label for i in train.examples]))
print("词向量的维度是", vectors_dim, "分类的维度是", num_class)

SENTENCE1.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE1.vocab.vectors.unk_init = init.xavier_uniform

SENTENCE2.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE2.vocab.vectors.unk_init = init.xavier_uniform

batch_size = 256

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
# writer.add_graph(model, [torch.from_numpy(np.random.randint(100, size=(128, 38))).long(),
#                          torch.from_numpy(np.random.randint(100, size=(128, 41))).long()])
print(SENTENCE1.vocab.vectors.shape)
model.embedding1.weight.data.copy_(SENTENCE1.vocab.vectors)
model.embedding2.weight.data.copy_(SENTENCE2.vocab.vectors)
model.to(DEVICE)

crition = F.cross_entropy
# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # ,lr=0.000001)
n_epoch = 10

best_val_acc = 0

writer = SummaryWriter()


def training(model, n_epoch, train_iter):
    for epoch in range(n_epoch):
        epoch_loss = 0
        train_acc = 0
        # Example object has no attribute sentence2，看前面 assert 那个
        for epoch2, batch in enumerate(train_iter):
            target = batch.gold_label
            # target.shape == 128
            target = target.to(DEVICE)
            optimizer.zero_grad()
            sentence1 = batch.sentence1_parse
            # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
            sentence1 = sentence1.permute(1, 0)
            sentence2 = batch.sentence2_parse
            sentence2 = sentence2.permute(1, 0)

            out = model(sentence1, sentence2)
            loss = crition(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.data
            train_acc += (torch.argmax(out, dim=-1) == target).sum().item()
        writer.add_scalar("Loss/epoch", epoch_loss / len(train), epoch)
        writer.add_scalar("acc/epoch", train_acc / len(train), epoch)
        print('epoch is ', epoch, "epoch_loss is", epoch_loss / len(train), "acc is", train_acc / len(train))


if __name__ == '__main__':
    training(model, 20, train_iter)
    writer.close()
