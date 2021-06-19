import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'esim_cnn'
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.out_features = out_features
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dropout = 0.5
        self.embedding = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim)
        self.lstm = nn.LSTM(input_size=config.embedding_dim, hidden_size=config.hidden_size, num_layers=2,
                            bidirectional=True,
                            batch_first=True)
        self.lstm2 = nn.LSTM(input_size=8 * config.hidden_size, hidden_size=config.hidden_size, num_layers=2,
                             bidirectional=True,
                             batch_first=True)
        # self.linear1 = nn.Linear(in_features=20 * 2, out_features=40)
        # self.linear2 = nn.Linear(in_features=20 * 2, out_features=2)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(config.hidden_size * 8),
            nn.Linear(config.hidden_size * 8, config.out_features),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.out_features),
            nn.Dropout(self.dropout),
            nn.Linear(config.out_features, config.out_features),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(config.out_features),
            nn.Dropout(config.dropout),
            nn.Linear(config.out_features, config.out_features),
            nn.Softmax(dim=-1)
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # premise_embedding, hypothesis_embedding = self.embedding(premise, hypothesis)
        a_bar, b_bar = self.input_encoding(a, b)
        a_hat, b_hat = self.inference_modeling(a_bar, b_bar)
        v = self.inference_composition(a_hat, b_hat, a_bar, b_bar)
        result = self.prediction(v)
        return result

    def input_encoding(self, a: torch.Tensor, b: torch.Tensor):
        # input: batch_size,seq_num
        a_embedding = self.embedding(a)
        b_embedding = self.embedding(b)
        # output: batch_size,seq_num,embedding_dim
        # 这里两个用的是同一个lstm, 而且是双向的lstm
        a_bar, (a0, a1) = self.lstm(a_embedding)
        b_bar, (b0, b1) = self.lstm(b_embedding)
        # output:batch_size, seq_num,  2 * hidden_size
        return a_bar, b_bar

    def inference_modeling(self, a_bar: torch.Tensor, b_bar: torch.Tensor):
        e_ij = torch.matmul(a_bar, b_bar.permute(0, 2, 1))
        # output： batch_size,seq_num_a,seq_num_b
        attention_a = F.softmax(e_ij)
        attention_b = F.softmax(e_ij.permute(0, 2, 1))
        a_hat = torch.matmul(attention_a, b_bar)
        b_hat = torch.matmul(attention_b, a_bar)
        # output: batch_size, seq_num, 2 * hidden_size
        return a_hat, b_hat

    def inference_composition(self, a_hat: torch.Tensor, b_hat: torch.Tensor, a_bar: torch.Tensor, b_bar: torch.Tensor):
        a_diff = a_bar - a_hat
        a_mul = torch.mul(a_bar, a_hat)
        m_a = torch.cat((a_bar, a_hat, a_diff, a_mul), dim=-1)
        # output: batch_size, seq_num_a, 2 * hidden_size * 4
        b_diff = b_bar - b_hat
        b_mul = torch.mul(b_bar, b_hat)
        m_b = torch.cat((b_bar, b_hat, b_diff, b_mul), dim=-1)
        # output: batch_size, seq_num_b, 2 * hidden_size * 4
        v_a, _ = self.lstm2(m_a)
        v_b, _ = self.lstm2(m_b)
        # output: batch_size, seq_num_b, 2 * hidden_size
        # v_a_mean = learn_torch.mean(v_a, dim=1)
        # v_b_mean = learn_torch.mean(v_b, dim=1)
        #
        # v_a_max = learn_torch.max(v_a, dim=1)
        # v_b_max = learn_torch.max(v_b, dim=1)
        #
        # v = learn_torch.cat((v_a_mean, v_a_max, v_b_mean, v_b_max), dim=1)
        # output:  batch_size,2 * seq_num_a + 2* seq_num_b, 2 * hidden
        q1_rep = self.apply_multiple(v_a)
        q2_rep = self.apply_multiple(v_b)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        return x

    def prediction(self, v: torch.Tensor):
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
    num_embeddings = 5000
    batch_size = 128
    max_length = 32
    embedding_dim = 300
    out_features = 10
    epoch = 5
    config = Config(num_embeddings, embedding_dim, out_features)
    model = Model(config)
    print(model)
    print("---------------------")
    for param in model.named_parameters():
        print(param[0], param[1].shape)
    inputs1 = torch.randint(high=200, size=(batch_size, max_length))
    inputs2 = torch.randint(high=200, size=(batch_size, max_length))
    res = (model(inputs1, inputs2))
    assert res.shape == (batch_size, out_features)
