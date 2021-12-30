import torch.nn as nn
import torch.nn.functional as F
import torch


class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'text_cnn2'
        self.kernel_size1 = 2
        self.kernel_size2 = 3
        self.kernel_size3 = 4  # 卷积核尺寸
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
        self.embedding = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(config.kernel_size1, config.embedding_dim))
        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(config.kernel_size2, config.embedding_dim))
        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(config.kernel_size3, config.embedding_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(in_features=1, out_features=config.out_features)

    def forward(self, input):
        #  (batch_size,seq_length)
        input_embedding = self.embedding(input)
        # 增加input channel
        input_embedding = input_embedding.unsqueeze(1)
        #  (batch_size,1,seq_length,embedding_dim)
        x1 = self.conv_and_pool(input_embedding, self.cnn1)
        x2 = self.conv_and_pool(input_embedding, self.cnn2)
        x3 = self.conv_and_pool(input_embedding, self.cnn3)
        # (batch_size, 1)
        x = torch.cat((x1, x2, x3), dim=0)
        # (3 *batch_size, 1)
        out = self.dropout(x)
        print(out.shape)
        out = self.fc(out)
        return out

    def conv_and_pool(self, x, conv):
        # x: batch_size,1,seq_length,embedding_dim
        # conv1: 1, 1,(kernel,embedding_dim)

        # conv(x): batch_size, 1, seq_length - kernel_size + 1, 1
        # F.relu(conv(x)): batch_size, 1, seq_length - kernel_size + 1, 1
        # F.relu(conv(x)).squeeze(-1): batch_size, 1, seq_length - kernel_size + 1
        # F.max_pool1d(x, x.size(-1)): batch_size, 1, 1

        # result: batch_size, 1
        x = F.relu(conv(x)).squeeze(-1)
        x = F.max_pool1d(x, x.size(-1)).squeeze(-1)
        return x

#
# if __name__ == '__main__':
#     num_embeddings = 5000
#     batch_size = 128
#     seq_length = 5
#     embedding_dim = 300
#     out_features = 10
#     config = Config(num_embeddings, embedding_dim, out_features)
#     model = Model(config)
#     print(model)
#     print("---------------------")
#     for param in model.named_parameters():
#         print(param[0], param[1].shape)
#     inputs1 = torch.randint(high=200, size=(batch_size, seq_length))
#     # 增加inputchannel 维度
#     inputs1 = inputs1.squeeze(1)
#     print(model(inputs1))
