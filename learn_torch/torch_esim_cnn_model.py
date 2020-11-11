import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Esim(nn.Module):
    def __init__(self, sentence1_vocab, sentence2_vocab, embedding_dim, hidden_size, num_class):
        super().__init__()
        self.dropout = 0.5
        self.embedding1 = nn.Embedding(num_embeddings=sentence1_vocab, embedding_dim=embedding_dim)
        self.embedding2 = nn.Embedding(num_embeddings=sentence2_vocab, embedding_dim=embedding_dim)
        # self.embedding2 = nn.Embedding(num_embeddings=sentence1_vocab, embedding_dim=embedding_dim)
        self.cnn = nn.Conv1d(in_channels=embedding_dim, out_channels=int(embedding_dim / 2), kernel_size=2)
        self.cnn2 = nn.Conv1d(in_channels=int(embedding_dim / 2), out_channels=int(embedding_dim / 2), kernel_size=2)
        # self.linear1 = nn.Linear(in_features=20 * 2, out_features=40)
        # self.linear2 = nn.Linear(in_features=20 * 2, out_features=2)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 8),
            nn.Linear(hidden_size * 8, num_class),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_class),
            nn.Dropout(self.dropout),
            nn.Linear(num_class, num_class),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(num_class),
            nn.Dropout(self.dropout),
            nn.Linear(num_class, num_class),
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
        a_embedding = a_embedding.permute(0, 2, 1)
        b_embedding = b_embedding.permute(0, 2, 1)
        # output: batch_size,embedding_dim,seq_num
        a_bar = self.cnn(a_embedding)
        # output：batch_size,out_channels(embedding_dim/2),  seq_num_a - kernel_size + 1
        b_bar = self.cnn(b_embedding)
        # output: batch_size,out_channels(embedding_dim/2),  seq_num_b - kernel_size + 1
        return a_bar, b_bar

    def inference_modeling(self, a_bar, b_bar):
        e_ij = torch.matmul(a_bar.permute(0, 2, 1), b_bar)
        # output： batch_size,seq_num_a-kernel_size+1 ,seq_num_b - kernel_size + 1
        attention_a = F.softmax(e_ij)
        attention_b = F.softmax(e_ij.permute(0, 2, 1))
        # batch_size, seq_num_a - kernel_size + 1, seq_num_b - kernel_size + 1 * (batch_size,out_channels(embedding_dim/2),  seq_num_b - kernel_size + 1)
        a_hat = torch.matmul(attention_a, b_bar.permute(0, 2, 1))
        b_hat = torch.matmul(attention_b, a_bar.permute(0, 2, 1))
        # output: batch_size, seq_num_a-kernel_size+1 , out_channels
        a_hat = a_hat.permute(0, 2, 1)
        b_hat = b_hat.permute(0, 2, 1)
        # batch_size, out_channels, seq_num_a - kernel_size + 1
        return a_hat, b_hat

    def inference_composition(self, a_hat, b_hat, a_bar, b_bar):
        a_diff = a_bar - a_hat
        # batch_size, out_channels, seq_num_a - kernel_size + 1
        a_mul = torch.mul(a_bar, a_hat)
        m_a = torch.cat((a_bar, a_hat, a_diff, a_mul), dim=2)
        # output:  batch_size, out_channels, 4* (seq_num_a - kernel_size + 1)
        b_diff = b_bar - b_hat
        b_mul = torch.mul(b_bar, b_hat)
        m_b = torch.cat((b_bar, b_hat, b_diff, b_mul), dim=2)
        # output: batch_size, out_channels, 4* (seq_num_a - kernel_size + 1)
        v_a = self.cnn2(m_a)
        v_b = self.cnn2(m_b)
        # output: batch_size, out_channels, (4* (seq_num_a - kernel_size + 1))-kernel_size + 1
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
        p1 = F.avg_pool1d(x.transpose(1, 2), int(x.size(1))).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), int(x.size(1))).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)


if __name__ == '__main__':
    esim = Esim(200, 200, 100, 2, 2)
    print(esim(torch.from_numpy(np.random.randint(100, size=(128, 38))).long(),
               torch.from_numpy(np.random.randint(100, size=(128, 41))).long()))
