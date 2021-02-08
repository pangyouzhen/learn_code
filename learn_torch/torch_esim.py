import datetime
import json
from typing import Callable, List

import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from torch.nn import init
from torch.utils.tensorboard import SummaryWriter
from torchtext import data
from torchtext.vocab import Vectors

from learn_torch.DataFrameDataSet import DataFrameDataset
from learn_torch.torch_esim_model import Esim


def tokenizer(text: str) -> List:
    return [tok for tok in text]


def tokenizer2(text: str) -> List:
    return text.split()

# TODO 这个esim效果为什么这么差
def init_text_match_data(df: pd.DataFrame, tokenizer: Callable, batch_size: int, vectors_name: str, vectors_path: str,
                         device=torch.device("cpu")):
    assert set(list(df)) == {'label', 'sentence1', 'sentence2'}
    LABEL: data.Field = data.Field(sequential=False, use_vocab=False)
    SENTENCE: data.Field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    train: DataFrameDataset = DataFrameDataset(df,
                                               fields={'sentence1': SENTENCE, 'sentence2': SENTENCE, "label": LABEL})
    # 使用本地词向量
    # torchtext.Vectors 会自动识别 headers
    vectors: Vectors = Vectors(name=vectors_name, cache=vectors_path)
    # 获取词向量的维度
    vectors_dim: int = vectors.dim
    # 获取分类的维度
    num_class = len(set([i.label for i in train.examples]))
    print("词向量的维度是", vectors_dim, "分类的维度是", num_class)
    SENTENCE.build_vocab(train, vectors=vectors)  # , max_size=30000)
    # 这里SENTENCE 根据vectors 构建了自己的词向量 ->> SENTENCE.vocab.vectors
    # 如果原始词向量中没有这个词-> 构建成一个0 tensor
    # 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
    SENTENCE.vocab.vectors.unk_init = init.xavier_uniform
    train_iter: data.BucketIterator = data.BucketIterator(train, batch_size=batch_size,
                                                          shuffle=True, device=device)
    # train_iter = data.BucketIterator(train, batch_size=batch_size, device=DEVICE)
    return SENTENCE, train_iter, num_class, train


# print(df[:5])
# print(df["label"].value_counts())


# print(SENTENCE1.vocab.vectors.shape) learn_torch.Size([1443, 300])
# 查看一个batch
# batch = next(iter(train_iter))
# print(batch.fields)
# print("batch text: ", batch.sentence1.shape) -> [64, 128] (seq_num_a,batch_size)
# print("batch text: ", batch.sentence2.shape) -> [63, 128]
# print("batch label: ", batch.label.shape) -> [128]


def init_model(SENTENCE: data.Field, num_class: int, device: torch.device,
               writer):
    model = Esim(sentence_vocab=len(SENTENCE.vocab), embedding_dim=SENTENCE.vocab.vectors.shape[-1], hidden_size=20,
                 num_class=num_class)
    # writer.add_graph(model, [torch.from_numpy(np.random.randint(100, size=(128, 38))).long(),
    #                          torch.from_numpy(np.random.randint(100, size=(128, 41))).long()])
    print(SENTENCE.vocab.vectors.shape)
    model.embedding.weight.data.copy_(SENTENCE.vocab.vectors)
    model.to(device)
    return model


def training(model: Esim, n_epoch: int, train_iter, device, train, writer, lr=0.05):
    logger.info("lr is {}, n_epoch is {}".format(lr, n_epoch))
    crition = F.cross_entropy
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # ,lr=0.000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    # with torchsnooper.snoop():
    for epoch in range(n_epoch):
        train_epoch_loss = 0
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
            train_epoch_loss = train_epoch_loss + loss.data
            # print((torch.argmax(out, dim=-1) == target).sum())
            train_acc += (torch.argmax(out, dim=-1) == target).sum().item()
        writer.add_scalar("train_loss/epoch", train_epoch_loss / len(train), epoch)
        writer.add_scalar("train_acc/epoch", train_acc / len(train), epoch)
        scheduler.step()
        logger.info("epoch is {} train_epoch_loss is {} train_acc is {}".format(epoch, train_epoch_loss,
                                                                                train_acc / len(train)))


def testing(model: Esim, n_epoch: int, test_iter: data.BucketIterator, device: torch.device, test: DataFrameDataset,
            writer, lr: float = 0.00001):
    crition = F.cross_entropy
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # ,lr=0.000001)
    for epoch in range(n_epoch):
        test_epoch_loss = 0
        test_acc = 0
        # Example object has no attribute sentence2，看前面 assert 那个
        for epoch2, batch in enumerate(test_iter):
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
            test_epoch_loss = test_epoch_loss + loss.data
            test_acc += (torch.argmax(out, dim=-1) == target).sum().item()
        writer.add_scalar("test_epoch_loss/epoch", test_epoch_loss / len(test), epoch)
        writer.add_scalar("test_acc/epoch", test_acc / len(test), epoch)
        logger.info(
            "epoch is {} test_epoch_loss is {} test_acc is {}".format(epoch, test_epoch_loss, test_acc / len(test)))
    torch.save(model, "./pkl/%s_esim.pkl" % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))


def main():
    # quotechar 参数，比如句子中只有一个",数据错乱的情况下
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)
    writer = SummaryWriter()
    n_epoch = 10
    batch_size: int = 256
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    logger.add("./log/%s.log" % datetime_now)
    train_df = pd.read_csv("../full_data/ants/ants_torchtext_train.csv", sep=",", encoding="utf-8")
    SENTENCE, train_iter, num_class, train = \
        init_text_match_data(train_df, tokenizer, batch_size,
                             vectors_name="sgns.sogounews.bigram-char",
                             vectors_path="../data/", device=device)
    model = init_model(SENTENCE, num_class, device, writer)
    training(model, n_epoch, train_iter, device, train, writer)
    # 验证集
    test_df = pd.read_csv("../full_data/ants/ants_torchtext_dev.csv", sep=",", encoding="utf-8")
    SENTENCE1, test_iter, num_class, test = \
        init_text_match_data(test_df, tokenizer, batch_size,
                             vectors_name="sgns.sogounews.bigram-char",
                             vectors_path="../data/", device=device)
    testing(model, n_epoch, test_iter, device, test, writer)
    writer.close()


def process_dataset(file: str):
    def split_multi_columns(x):
        js = json.loads(x["json"])
        return js["sentence1"], js["sentence2"], js["label"]

    df = pd.read_csv(file, sep="\t", encoding="utf-8", names=["json"])
    # 别忘了result_type 参数
    df[['sentence1', 'sentence2', 'label']] = df.apply(lambda x: split_multi_columns(x), axis=1, result_type="expand")
    df = df[['sentence1', 'sentence2', 'label']]
    df.to_csv("../full_data/ants/ants_torchtext_dev.csv", index=False, encoding="utf-8")


if __name__ == '__main__':
    main()
# 运行tensorboard
# cd /data/project/learn_code/learn_torch
# tensorboard --logdir=./runs
#  以后要写成函数或者类的形式，这样的好处是可以单独对某一个函数进行测试，而不是将整个程序都重新跑一遍
#  要善于使用try 和 assert函数
# pd.DataFrame({"target": target.cpu().numpy(),"out":out.cpu().detach().numpy(),"argmax":torch.argmax(out,dim=-1).cpu().numpy()})
