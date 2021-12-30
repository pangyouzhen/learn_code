import torch
import torch.nn as nn

a = torch.randint(10, size=(5, 4))
embedding = nn.Embedding(num_embeddings=100, embedding_dim=10)
embed = embedding(a)

lstm = nn.LSTM(input_size=10, hidden_size=5, num_layers=3, batch_first=True)
output, (h_n, c_n) = lstm(embed)
for param in lstm.named_parameters():
    # print(param,param.size())
    print(param[0], "++", param[1].shape)
print(output.shape)
print(h_n.shape)
print(c_n.shape)
