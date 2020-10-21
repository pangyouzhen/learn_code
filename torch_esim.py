import json

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchtext import data
from torchtext.vocab import Vectors

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

train = data.TabularDataset('./full_data/ants/ants_torchtext_train.csv', format='tsv', skip_header=True,
                            fields=[('sentence1', SENTENCE1), ('sentence2', SENTENCE2), ('label', LABEL)])

# 使用本地词向量
# torchtext.Vectors 会自动识别 headers
vectors = Vectors(name="glove.6B.100d.txt", cache="/data/project/learn_allennlp/data/.vector_cache/")
SENTENCE1.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE1.vocab.vectors.unk_init = init.xavier_uniform
#  第1510 个词
print(SENTENCE1.vocab.itos[1510])

train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.Phrase),
                                 shuffle=True, device=DEVICE)
sentence1_vocab = len(SENTENCE1.vocab)


# 文字-index-embeding
class Esim(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=sentence1_vocab, embedding_dim=embedding_dim)
        # self.embedding2 = nn.Embedding(num_embeddings=sentence1_vocab, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=20, num_layers=2, bidirectional=True)
        self.linear1 = nn.Linear(in_features=20 * 2, out_features=40)
        self.linear2 = nn.Linear(in_features=20 * 2, out_features=2)

    def forward(self, a, b):
        # premise_embedding, hypothesis_embedding = self.embedding(premise, hypothesis)
        a_bar, b_bar = self.input_encoding(a, b)
        a_hat, b_hat = self.inference_modeling(a_bar, b_bar)
        v = self.inference_composition(a_hat, b_hat, a_bar, b_bar)
        result = self.prediction(v)
        return result

    def input_encoding(self, a, b):
        # input: str,str
        a_embedding = self.embedding1(a)
        b_embedding = self.embedding1(b)
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
        b_diff = b_bar - b_hat
        b_mul = torch.mul(b_bar, b_hat)
        m_b = torch.cat((b_bar, b_hat, b_diff, b_mul), dim=2)

        v_a = self.lstm(m_a)
        v_b = self.lstm(m_b)
        v_a_mean = torch.mean(v_a, dim=1)
        v_b_mean = torch.mean(v_b, dim=1)

        v_a_max = torch.max(v_a, dim=1)
        v_b_max = torch.max(v_b, dim=1)

        v = torch.cat((v_a_mean, v_a_max, v_b_mean, v_b_max), dim=1)
        # output:  batch_size,2 * seq_num_a + 2* seq_num_b, 2 * hidden
        return v

    # TODO
    def prediction(self, v):
        # batch_size, 2 * seq_num_a + 2 * seq_num_b, 2 * hidden
        feed_forward1 = self.linear1(v)
        # batch_size,2 * seq_num_a + 2 * seq_num_b,  2 * hidden
        tanh_layer = F.tanh(feed_forward1)
        feed_forward2 = self.linear2(tanh_layer)
        softmax = F.softmax(feed_forward2)
        return softmax


model = Esim(300)
# TODO ERROR
# the size of tensor a (300) must match  the size of tensor b (100) at non-singeton
model.embedding.weight.data.copy_(SENTENCE1.vocab.vectors)
model.to(DEVICE)

model.to(DEVICE)

# 训练
optimizer = torch.optim.Adam(model.parameters())  # ,lr=0.000001)

n_epoch = 20

best_val_acc = 0

for epoch in range(n_epoch):

    for batch_idx, batch in enumerate(train_iter):
        # 124849 / 128 batch_size -> 975 batch
        data = batch.Phrase
        # type(data) == Tensor
        # data.shape == (...==seq_num,128)
        # print("shape data is %s %s %s" % (batch_idx, data.shape[0], data.shape[1]))
        target = batch.Sentiment
        # 这里的目的是什么？
        # target.shape == 128
        target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        # 这里data 为什么进行转换
        # Adam 和 SGD的区别是什么
        data = data.permute(1, 0)
        optimizer.zero_grad()

        out = model(data)
        print("---------------------------")
        print(out.shape)
        loss = -target * torch.log(out) - (1 - target) * torch.log(1 - out)
        loss = loss.sum(-1).mean()

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 200 == 0:
            _, y_pre = torch.max(out, -1)
            acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
            print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f'
                  % (epoch, batch_idx, loss, acc))

        if batch_idx > 974:
            _, y_pre = torch.max(out, -1)
            acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
            print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f'
                  % (epoch, batch_idx, loss, acc))

    # val_accs = []
    # for batch_idx, batch in enumerate(val_iter):
    #     data = batch.Phrase
    #     target = batch.Sentiment
    #     target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
    #     target = target.to(DEVICE)
    #     data = data.permute(1, 0)
    #     out = model(data)
    #
    #     _, y_pre = torch.max(out, -1)
    #     acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
    #     val_accs.append(acc)
    #
    # acc = np.array(val_accs).mean()
    # if acc > best_val_acc:
    #     print('val acc : %.4f > %.4f saving model' % (acc, best_val_acc))
    #     torch.save(model.state_dict(), 'params.pkl')
    #     best_val_acc = acc
    # print('val acc: %.4f' % (acc))
