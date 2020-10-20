import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 这个用torchtext 比较麻烦
sougou_vector = gensim.models.KeyedVectors.load_word2vec_format("./data/sgns.sogounews.bigram-char")

print(sougou_vector.vocab["的"].index)
embedding_dim = sougou_vector.vocab["的"].size

df = pd.read_csv("./full_data/ants/train.json", sep="\t", encoding="utf-8", names=["json"])


def split_multi_columns(x):
    js = json.loads(x["json"])
    return js["sentence1"], js["sentence2"], js["label"]


def get_vector_index(x):
    return np.array([sougou_vector.vocab[i].index for i in x if i in sougou_vector.vocab.keys()])


# 别忘了result_type 参数
df[['sentence1', 'sentence2', 'label']] = df.apply(lambda x: split_multi_columns(x), axis=1, result_type="expand")

# 这里没必要转换成index，直接获取embedding
# df["sentence1_index"] = df["sentence1"].apply(get_vector_index)
# df["sentence2_index"] = df["sentence2"].apply(get_vector_index)


batch_size = 3


# 文字-index-embeding
class Esim(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=20, num_layers=2, bidirectional=True)
        self.linear = nn.Linear(in_features=batch_size * 4)

    def forward(self, a, b):
        # premise_embedding, hypothesis_embedding = self.embedding(premise, hypothesis)
        a_bar, b_bar = self.input_encoding(a, b)
        a_hat, b_hat = self.inference_modeling(a_bar, b_bar)
        v = self.inference_composition(a_hat, b_hat, a_bar, b_bar)
        result = self.prediction(v)
        return result

    def input_encoding(self, a, b):
        # input: str,str
        a_embeding, b_embedding = self.init_embedding(a, b)
        # output: batch_size,seq_num,embedding_dim
        a_bar, (a0, a1) = self.lstm(a_embeding)
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
        return v

    def prediction(self, v):
        feed_forward1 = self.linear(v)
        F.tanh(feed_forward1)

    def init_embedding(self, a, b):
        return np.array([sougou_vector[i] for i in a if i in sougou_vector.vocab.keys()]), np.array(
            [sougou_vector[i] for i in b if i in sougou_vector.vocab.keys()])


if __name__ == '__main__':
    pass
    # a = torch.randn(2, 500)
    # b = torch.randn(2, 500)
