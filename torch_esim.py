import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors

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


def tokenizer(text):
    return [tok for tok in text]


LABEL = data.Field(sequential=False, use_vocab=False)
SENTENCE1 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
SENTENCE2 = data.Field(sequential=True, tokenize=tokenizer, lower=True)

train = data.TabularDataset('./full_data/ants/ants_torchtext_train.csv', format='csv', skip_header=True,
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
sentence2_vocab = len(SENTENCE2.vocab)


# print(SENTENCE1.vocab.vectors.shape) torch.Size([1443, 300])
# 查看一个batch
# batch = next(iter(train_iter))
# print(batch.fields)
# print("batch text: ", batch.sentence1.shape) -> [64, 128] (seq_num_a,batch_size)
# print("batch text: ", batch.sentence2.shape) -> [63, 128]
# print("batch label: ", batch.label.shape) -> [128]

class Esim(nn.Module):
    def __init__(self, sentence1_vocab, embedding_dim, hidden_size):
        super().__init__()
        self.dropout = 0.5
        self.embedding1 = nn.Embedding(num_embeddings=sentence1_vocab, embedding_dim=embedding_dim)
        self.embedding2 = nn.Embedding(num_embeddings=sentence2_vocab, embedding_dim=embedding_dim)
        # self.embedding2 = nn.Embedding(num_embeddings=sentence1_vocab, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=8 * hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        # self.linear1 = nn.Linear(in_features=20 * 2, out_features=40)
        # self.linear2 = nn.Linear(in_features=20 * 2, out_features=2)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 8),
            nn.Linear(hidden_size * 8, 2),
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

    def forward(self, a, b):
        # premise_embedding, hypothesis_embedding = self.embedding(premise, hypothesis)
        a_bar, b_bar = self.input_encoding(a, b)
        a_hat, b_hat = self.inference_modeling(a_bar, b_bar)
        v = self.inference_composition(a_hat, b_hat, a_bar, b_bar)
        result = self.prediction(v)
        return result

    def input_encoding(self, a, b):
        # input: batch_size,seq_num
        a_embedding = self.embedding1(a)
        b_embedding = self.embedding2(b)
        # output: batch_size,seq_num,embedding_dim
        a_bar, (a0, a1) = self.lstm(a_embedding)
        b_bar, (b0, b1) = self.lstm(b_embedding)
        # output: batch_size,seq_num,2 * hidden_size
        return a_bar, b_bar

    def inference_modeling(self, a_bar, b_bar):
        e_ij = torch.matmul(a_bar, b_bar.permute(0, 2, 1))
        # output： batch_size,seq_num_a,seq_num_b
        attention_a = F.softmax(e_ij)
        attention_b = F.softmax(e_ij.permute(0, 2, 1))
        a_hat = torch.matmul(attention_a, b_bar)
        b_hat = torch.matmul(attention_b, a_bar)
        # output: batch_size, seq_num, 2 * hidden_size
        return a_hat, b_hat

    def inference_composition(self, a_hat, b_hat, a_bar, b_bar):
        a_diff = a_bar - a_hat
        a_mul = torch.mul(a_bar, a_hat)
        m_a = torch.cat((a_bar, a_hat, a_diff, a_mul), dim=2)
        # output: batch_size, seq_num_a, 2 * hidden_size * 4
        b_diff = b_bar - b_hat
        b_mul = torch.mul(b_bar, b_hat)
        m_b = torch.cat((b_bar, b_hat, b_diff, b_mul), dim=2)
        # output: batch_size, seq_num_b, 2 * hidden_size * 4
        v_a, _ = self.lstm2(m_a)
        v_b, _ = self.lstm2(m_b)
        # output: batch_size, seq_num_b, 2 * hidden_size
        # v_a_mean = torch.mean(v_a, dim=1)
        # v_b_mean = torch.mean(v_b, dim=1)
        #
        # v_a_max = torch.max(v_a, dim=1)
        # v_b_max = torch.max(v_b, dim=1)
        #
        # v = torch.cat((v_a_mean, v_a_max, v_b_mean, v_b_max), dim=1)
        # output:  batch_size,2 * seq_num_a + 2* seq_num_b, 2 * hidden
        q1_rep = self.apply_multiple(v_a)
        q2_rep = self.apply_multiple(v_b)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        return x

    def prediction(self, v):
        # batch_size, 2 * seq_num_a + 2 * seq_num_b, 2 * hidden
        # feed_forward1 = self.linear1(v)
        # batch_size,2 * seq_num_a + 2 * seq_num_b,  2 * hidden
        # tanh_layer = F.tanh(feed_forward1)
        # feed_forward2 = self.linear2(tanh_layer)
        # softmax = F.softmax(feed_forward2)
        score = self.fc(v)
        return score

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)


model = Esim(sentence1_vocab=sentence1_vocab, embedding_dim=300, hidden_size=20)
print(SENTENCE1.vocab.vectors.shape)
model.embedding1.weight.data.copy_(SENTENCE1.vocab.vectors)
model.embedding2.weight.data.copy_(SENTENCE2.vocab.vectors)
model.to(DEVICE)

crition = F.cross_entropy
# 训练
optimizer = torch.optim.Adam(model.parameters())  # ,lr=0.000001)

n_epoch = 20

best_val_acc = 0

for epoch in range(n_epoch):
    epoch_loss = 0
    acc = 0
    # Example object has no attribute sentence2，看前面 assert 那个
    for epoch2, batch in enumerate(train_iter):
        # 124849 / 128 batch_size -> 975 batch
        # type(data) == Tensor
        # data.shape == (...==seq_num,128)
        # print("shape data is %s %s %s" % (batch_idx, data.shape[0], data.shape[1]))
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
        # print("---------------------------")
        # print(out.shape)
        loss = crition(out, target)
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.data
        y_pre = torch.argmax(out, dim=-1)
        acc = torch.mean((torch.tensor(y_pre == batch.label, dtype=torch.float)))
    print("epoch_loss is", epoch_loss, "acc is", acc)
