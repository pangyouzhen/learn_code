import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets
from torchtext.vocab import Vectors
import pandas as pd
import jieba
import gensim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def tokenizer(text):  # create a tokenizer function
#     """
#     定义分词操作
#     """
#     return [tok.text for tok in jieba.tokenizer(text)]
# class SogouVectors(Vectors):
#
#     def __init__(self, path):
#         super(SogouVectors, self).__init__(path)
#         with open(path, encoding="utf-8") as f:
#             first_line = next(f)
#             self.vectors = f.readlines()
#
#     def __getitem__(self, item):
#         return self.vectors[item]
#
#     def __len__(self):
#         return len(self.vectors)
#
#
# sougou_vector = SogouVectors("./data/sgns.sogounews.bigram-char")
sougou_vector = gensim.models.KeyedVectors.load_word2vec_format(r'PubMed-and-PMC-w2v.bin', binary=True)
"""
field在默认的情况下都期望一个输入是一组单词的序列，并且将单词映射成整数。
这个映射被称为vocab。如果一个field已经被数字化了并且不需要被序列化，
可以将参数设置为use_vocab=False以及sequential=False。
"""
LABEL = data.Field(sequential=False, use_vocab=False)

TEXT = data.Field(sequential=True, lower=True)
# TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)

train, val = data.TabularDataset.splits(
    path='.', train='./full_data/train.csv', validation='./full_data/val.csv', format='csv', skip_header=True,
    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)])

test = data.TabularDataset('./full_data/test.tsv', format='tsv', skip_header=True,
                           fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)])


# Vectors 将每一个字都变成数字
class Esim(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(input_size=500, hidden_size=20, num_layers=2, bidirectional=True)

    def forward(self, a, b):
        # premise_embedding, hypothesis_embedding = self.embedding(premise, hypothesis)
        a_bar, b_bar = self.input_encoding(a, b)
        m_a, m_b = self.inference_modeling(a_bar, b_bar)
        V = self.inference_composition(m_a, m_b)
        result = self.prediction(V)
        return result

    def input_encoding(self, a, b):
        a_embeding, b_embedding = self.embedding(a, b)

    #    (batch_size,)

    def inference_modeling(self, a_bar, b_bar):
        pass

    def inference_composition(self, m_a, m_b):
        pass

    def prediction(self, a, b):
        pass


if __name__ == '__main__':
    a = torch.randn(2, 500)
    b = torch.randn(2, 500)
