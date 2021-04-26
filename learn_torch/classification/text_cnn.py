import torch.nn as nn


# TODO
class TextCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, a, b):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.cnn = nn.Conv1d(in_channels=a, out_channels=b, kernel_size=a)
        self.b = b

    def forward(self, input):
        #  (batch_size,seq_length)
        input_embedding = self.embedding(input)
        # (batch_size,seq_length,embedding)
        cnn_embedding = self.cnn(input_embedding)
