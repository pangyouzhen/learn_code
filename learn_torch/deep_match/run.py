import json
from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

vocab_size = 5000
batch_size = 128
max_length = 32
embed_dim = 300
label_num = 2
epoch = 20
train_path = '../../full_data/ants/train.json'
dev_path = '../../full_data/ants/train.json'
vocab_path = '/data/project/nlp_summary/data/THUCNews/data/vocab.txt'

output_path = './output'


def get_data(path):
    vocab_df = pd.read_csv(vocab_path, sep="\t", names=["word", "index"])
    vocabs = {val["word"]: int(val["index"]) for ind, val in vocab_df.iterrows()}
    vocabs_keys = vocabs.keys()

    def word2ind(x):
        a = [0] * max_length
        for i, v in enumerate(x):
            if i < max_length:
                if v in vocabs_keys:
                    a[i] = vocabs[v]
                else:
                    a[i] = 1
        return a

    def parser_json(x: pd.DataFrame):
        t = json.loads(x["all_columns"])
        return t["sentence1"], t["sentence2"], t["label"]

    df: pd.DataFrame = pd.read_csv(path, sep="\t", names=["all_columns"])
    print(df[:5])
    df[["sentence1", "sentence2", "label"]] = df.apply(parser_json, result_type="expand", axis=1)
    df["label"] = df["label"].astype(int)
    print(df.describe())
    df["vector1"] = df["sentence1"].apply(word2ind)
    df["vector2"] = df["sentence2"].apply(word2ind)
    x1 = np.array(df["vector1"].tolist())
    x2 = np.array(df["vector2"].tolist())
    y = np.array(df["label"].tolist())
    print(df["label"].value_counts())
    df["sentence1_len"] = df['sentence1'].apply(len)
    df["sentence2_len"] = df["sentence2"].apply(len)
    print(df.describe())
    label_num = len(set(df["label"].tolist()))
    return x1, x2, y, label_num


class DealDataset(Dataset):
    def __init__(self, x_train1, x_train2, y_train, device):
        self.x_data1 = torch.from_numpy(x_train1).long().to(device)
        self.x_data2 = torch.from_numpy(x_train2).long().to(device)
        self.y_data = torch.from_numpy(y_train).long().to(device)
        self.len = x_train1.shape[0]

    def __getitem__(self, index):
        return self.x_data1[index], self.x_data2[index], self.y_data[index]

    def __len__(self):
        return self.len


def evaluate(model, dataloader_dev):
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for sentence1, sentence2, labels in dataloader_dev:
            output = model(sentence1, sentence2)
            predic = torch.max(output.data, 1)[1].cpu()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, labels.cpu())
            if len(predict_all) > 1000:
                break
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc


def main(label_num):
    debug = False
    # 相对路径 + modelName(TextCNN、TextLSTM)
    model_name = 'abcnn'
    module = import_module(model_name)
    config = module.Config(vocab_size, embed_dim, label_num)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = module.Model(config).to(device)
    if debug:
        # 维度：batch_size * max_length, 数值：0~200之间的整数，每一行表示wordid
        inputs = torch.randint(0, 200, (batch_size, max_length))
        # 维度：batch_size * 1， 数值：0~2之间的整数，维度扩充1，和input对应
        labels = torch.randint(0, 2, (batch_size, 1)).squeeze(0)
        print(model(inputs))
    else:
        x1_train, x2_train, y_train, label_num = get_data(train_path)
        dataset = DealDataset(x1_train, x2_train, y_train, device)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        x_dev1, x_dev2, y_dev, _ = get_data(dev_path)
        dataset_dev = DealDataset(x_dev1, x_dev2, y_dev, device)
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        model.train()
        dev_acc = 0
        for i in range(epoch):
            index = 0
            for sentence1, sentence2, labels in tqdm(dataloader):
                model.zero_grad()
                output = model(sentence1, sentence2)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                index += 1
                if index % 50 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    true = labels.data.cpu()
                    predic = torch.max(output.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc = evaluate(model, dataloader_dev)
                    print(f'epoch:{i} batch:{index} loss:{loss} train_acc:{train_acc} dev_acc:{dev_acc}')
                    model.train()
        torch.save(model, f'{output_path}/{model_name}_{dev_acc}_{epoch}.pt')
        print('train finish')


if __name__ == "__main__":
    main(label_num)
    # get_data(train_path)
