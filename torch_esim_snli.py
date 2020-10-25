import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


model = Esim(sentence1_vocab=sentence1_vocab, embedding_dim=100, hidden_size=20)
print(SENTENCE1.vocab.vectors.shape)
model.embedding1.weight.data.copy_(SENTENCE1.vocab.vectors)
model.embedding2.weight.data.copy_(SENTENCE2.vocab.vectors)
model.to(DEVICE)

crition = F.cross_entropy
# 训练
optimizer = torch.optim.Adam(model.parameters())  # ,lr=0.000001)

n_epoch = 50

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
        # print(out.shape)
        loss = crition(out, target)
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.data
        y_pre = torch.argmax(out, dim=-1)
        train_acc += (torch.argmax(out, dim=-1) == target).sum().item()
    print("epoch_loss is", epoch_loss / len(train), "acc is", train_acc / len(train))
