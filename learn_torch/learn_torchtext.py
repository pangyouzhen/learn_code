import torch
import pandas as pd
from torchtext.data import Field

from utils.DataFrameDataSet import DataFrameDataset

# device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

train_df = pd.read_csv("../data/THUCNews/data/train.txt", sep="\t", names=["context", "class_"])
print(train_df[:5])


def tokenizer(text):
    return [tok for tok in text]


LABEL: Field = Field(sequential=False, use_vocab=False)
SENTENCE: Field = Field(sequential=True, tokenize=tokenizer, lower=True)

train: DataFrameDataset = DataFrameDataset(train_df,
                                           fields={'context': SENTENCE, "class_": LABEL})
print("examples___________________________start")
print(train.examples[0])
print(train.examples[0].class_)
print(train.examples[0].context)
print("examples___________________________end")
lr = 1e-3
num_class = len(set([i.class_ for i in train.examples]))
print("分类的维度是", num_class)
SENTENCE.build_vocab(train)
# SENTENCE.vocab.vectors.unk_init = init.xavier_uniform
print(SENTENCE.vocab.itos[:10])
# 默认情况下前两个 unk pad
print(SENTENCE)
print(type(SENTENCE))
print(dir(SENTENCE))
print(SENTENCE.vocab.stoi["中"])
print(SENTENCE.vocab.stoi["华"])
print(SENTENCE.numericalize(['中', '华', '女', '子', '学', '院', '：', '本', '科', '层', '次', '仅', '1', '专', '业', '招', '男', '生']))
print("field end_______________")
# train_iter: BucketIterator = BucketIterator(train, batch_size=128,
#                                             shuffle=True, device=device)
#
# vocab_size = 5000
# batch_size = 128
# max_length = 32
# embed_dim = 300
# label_num = 10
# epoch = 5
#
#
# class Config(object):
#     def __init__(self, vocab_size, embed_dim, label_num):
#         self.model_name = 'TextLSTM'
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.label_num = label_num
#         self.hidden_size = 128
#         self.num_layer = 2
#         self.dropout = 0.2
#         self.lr = 0.001
#
#
# class Model(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)
#         self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layer,
#                             bidirectional=True, batch_first=True, dropout=config.dropout)
#         self.fc = nn.Linear(config.hidden_size * 2, config.label_num)
#
#     def forward(self, input):
#         # input: batchsize,seq_length = 128, 50
#         embed = self.embedding(input)
#         # embed: batchsize,seq_length,embed_dim = 128, 50, 300
#         hidden, _ = self.lstm(embed)
#         # hidden: batchsize, seq, embedding = 128, 50, 256
#         hidden = hidden[:, -1, :]
#         # hidden: batchsize, seq_embedding = 128, 256
#         logit = torch.sigmoid(self.fc(hidden))
#         # logit: batchsize, label_logit = 128, 10
#         return logit
#
#
# config = Config(vocab_size, embed_dim, label_num)
# model = Model(config)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# model.train()
# best_acc = 0
# for i in range(epoch):
#     index = 0
#     for datas, labels in dataloader:
#         model.zero_grad()
#         output = model(datas)
#         loss = F.cross_entropy(output, labels)
#         loss.backward()
#         optimizer.step()
#         index += 1
#         if index % 50 == 0:
#             # 每多少轮输出在训练集和验证集上的效果
#             true = labels.data.cpu()
#             predic = torch.max(output.data, 1)[1].cpu()
#             train_acc = metrics.accuracy_score(true, predic)
#             dev_acc = evaluate(model, dataloader_dev)
#             print(f'epoch:{i} batch:{index} loss:{loss} train_acc:{train_acc} dev_acc:{dev_acc}')
#             # if dev_acc > best_acc:
#             #     torch.save(model, f'{output_path}/{model_name}/model.pt')
#             model.train()
#
# print('train finish')
if __name__ == '__main__':
    print(tokenizer("今天天气很好啊"))
