import torch.nn as nn


# TODO
class TextCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_channel, output_channel):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.cnn = nn.Conv1d(in_channels=input_channel, out_channels=output_channel, kernel_size=input_channel)
        self.outputchannel = output_channel

    def forward(self, input):
        #  (batch_size,seq_length)
        input_embedding = self.embedding(input)
        # (batch_size,seq_length,embedding)
        cnn_embedding = self.cnn(input_embedding)
