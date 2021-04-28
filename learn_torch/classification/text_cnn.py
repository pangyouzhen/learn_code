import torch.nn as nn


# todo
class Config(object):
    def __init__(self, num_embeddings, embedding_dim, out_features):
        self.model_name = 'text_cnn'
        self.kernel_size = 2
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
        self.cnn = nn.Conv1d(in_channels=config.input_channel, out_channels=config.output_channel,
                             kernel_size=config.kernel_size)

    def forward(self, input):
        #  (batch_size,seq_length)
        input_embedding = self.embedding(input)
        # (batch_size,seq_length,embedding) -> (N,C_in,L_in)
        cnn_encoding = self.cnn(input_embedding)
        # (batch_size,output_channel,)
