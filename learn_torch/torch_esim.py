# https://blog.csdn.net/qq_44722174/article/details/104640018

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
from torchtext import data
from torchtext.vocab import Vectors

from learn_torch.DataFrameDataSet import DataFrameDataset
from learn_torch.torch_esim_model import Esim


def tokenizer(text):
    return [tok for tok in text]


def init_text_match_data(df: pd.DataFrame, tokenizer, batch_size, vectors_name, vectors_path, DEVICE="cpu"):
    assert set(list(df)) == {'label', 'sentence1', 'sentence2'}
    LABEL = data.Field(sequential=False, use_vocab=False)
    SENTENCE1 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    SENTENCE2 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    train = DataFrameDataset(df, fields={'sentence1': SENTENCE1, 'sentence2': SENTENCE2, "label": LABEL})
    # 使用本地词向量
    # torchtext.Vectors 会自动识别 headers
    vectors = Vectors(name=vectors_name, cache=vectors_path)
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
    train_iter = data.BucketIterator(train, batch_size=batch_size,
                                     shuffle=True, device=DEVICE)
    return SENTENCE1, SENTENCE2, LABEL, train_iter, vectors_dim, num_class, train


# print(df[:5])
# print(df["label"].value_counts())


# print(SENTENCE1.vocab.vectors.shape) learn_torch.Size([1443, 300])
# 查看一个batch
# batch = next(iter(train_iter))
# print(batch.fields)
# print("batch text: ", batch.sentence1.shape) -> [64, 128] (seq_num_a,batch_size)
# print("batch text: ", batch.sentence2.shape) -> [63, 128]
# print("batch label: ", batch.label.shape) -> [128]


def init_model(SENTENCE1, SENTENCE2, vectors_dim, num_class, device):
    model = Esim(sentence1_vocab=len(SENTENCE1.vocab), sentence2_vocab=len(SENTENCE2.vocab), embedding_dim=vectors_dim,
                 hidden_size=20, num_class=num_class)
    # writer.add_graph(model, [torch.from_numpy(np.random.randint(100, size=(128, 38))).long(),
    #                          torch.from_numpy(np.random.randint(100, size=(128, 41))).long()])
    print(SENTENCE1.vocab.vectors.shape)
    model.embedding1.weight.data.copy_(SENTENCE1.vocab.vectors)
    model.embedding2.weight.data.copy_(SENTENCE2.vocab.vectors)
    model.to(device)
    return model


def training(model, n_epoch, train_iter, device, train, lr=0.01):
    crition = F.cross_entropy
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # ,lr=0.000001)
    for epoch in range(n_epoch):
        epoch_loss = 0
        train_acc = 0
        # Example object has no attribute sentence2，看前面 assert 那个
        for epoch2, batch in enumerate(train_iter):
            target = batch.label
            # target.shape == 128
            target = target.to(device)
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
        # writer.add_scalar("Loss/epoch", epoch_loss / len(train), epoch)
        # writer.add_scalar("acc/epoch", train_acc / len(train), epoch)
        print('epoch is ', epoch, "epoch_loss is", epoch_loss, "acc is", train_acc / len(train))


if __name__ == '__main__':
    # quotechar 参数，比如句子中只有一个",数据错乱的情况下
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    n_epoch = 20
    best_val_acc = 0
    batch_size = 128
    df = pd.read_csv("../full_data/ants/ants_torchtext_train.csv", sep=",", encoding="utf-8")
    SENTENCE1, SENTENCE2, LABEL, train_iter, vectors_dim, num_class, train = \
        init_text_match_data(df, tokenizer, batch_size,
                             vectors_name="sgns.sogounews.bigram-char",
                             vectors_path="../data/")
    model = init_model(SENTENCE1, SENTENCE2, vectors_dim, num_class, device=DEVICE)
    training(model, 20, train_iter, device=DEVICE, train=train)
    writer.close()
# 运行tensorboard
# cd /data/project/learn_code/learn_torch
# tensorboard --logdir=./runs
#  以后要写成函数或者类的形式，这样的好处是可以单独对某一个函数进行测试，而不是将整个程序都重新跑一遍
#  要善于使用try 和 assert函数
