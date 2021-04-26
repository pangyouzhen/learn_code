import torch
import torch.nn.functional as F
from torch.nn import init
from torchtext.vocab import Vectors

from learn_torch.deep_match.torch_esim_model import Esim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("--------------------------------")
print(DEVICE)


def tokenizer(text):
    return [tok for tok in text]


# 增加读取文件类型判断
# 使用本地词向量
# torchtext.Vectors 会自动识别 headers
vectors = Vectors(name="sgns.sogounews.bigram-char", cache="../data/")
# 获取词向量的维度
vectors_dim = vectors.dim
# 获取分类的维度
num_class = len(set([i.label for i in train.examples]))
print("词向量的维度是", vectors_dim, "分类的维度是", num_class)

SENTENCE1.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE1.vocab.vectors.unk_init = init.xavier_uniform

SENTENCE2.build_vocab(train, vectors=vectors)  # , max_size=30000)
# 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
SENTENCE2.vocab.vectors.unk_init = init.xavier_uniform

batch_size = 128

train_iter = data.BucketIterator(train, batch_size=batch_size,
                                 shuffle=True, device=DEVICE)
sentence1_vocab = len(SENTENCE1.vocab)
sentence2_vocab = len(SENTENCE2.vocab)

# print(SENTENCE1.vocab.vectors.shape) learn_torch.Size([1443, 300])
# 查看一个batch
# batch = next(iter(train_iter))
# print(batch.fields)
# print("batch text: ", batch.sentence1.shape) -> [64, 128] (seq_num_a,batch_size)
# print("batch text: ", batch.sentence2.shape) -> [63, 128]
# print("batch label: ", batch.label.shape) -> [128]


model = Esim(sentence1_vocab=sentence1_vocab, sentence2_vocab=sentence2_vocab, embedding_dim=vectors_dim,
             hidden_size=20, num_class=num_class)
print(SENTENCE1.vocab.vectors.shape)
model.embedding1.weight.data.copy_(SENTENCE1.vocab.vectors)
model.embedding2.weight.data.copy_(SENTENCE2.vocab.vectors)
model.to(DEVICE)

crition = F.cross_entropy
# 训练
optimizer = torch.optim.Adam(model.parameters())  # ,lr=0.000001)

n_epoch = 20

best_val_acc = 0

for epoch in range(n_epoch):
    epoch_loss = 0
    train_acc = 0
    # Example object has no attribute sentence2，看前面 assert 那个
    for epoch2, batch in enumerate(train_iter):
        target = batch.label
        # target.shape == 128
        target = target.to(DEVICE)
        optimizer.zero_grad()
        sentence1 = batch.sentence1
        # (seq_num_a,batch_size) -> (batch_size,seq_num_a)
        sentence1 = sentence1.permute(1, 0)
        sentence2 = batch.sentence2
        sentence2 = sentence2.permute(1, 0)

        out = model(sentence1, sentence2)
        loss = crition(out, target)
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.data
        train_acc += (torch.argmax(out, dim=-1) == target).sum().item()
    print("epoch_loss is", epoch_loss, "acc is", train_acc / len(train))
