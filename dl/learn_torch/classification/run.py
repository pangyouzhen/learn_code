from importlib import import_module

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
vocab_size = 5000
batch_size = 128
max_length = 32
embed_dim = 300
label_num = 10
epochs = 50
train_path = '/data/project/nlp_summary/data/THUCNews/data/train.txt'
dev_path = '/data/project/nlp_summary/data/THUCNews/data/dev.txt'
vocab_path = '/data/project/nlp_summary/data/THUCNews/data/vocab.txt'

output_path = 'output/'


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

    df = pd.read_csv(path, sep="\t", names=["sentence", "label"])
    # print(df.describe())
    print(df["label"].value_counts())
    df["vector"] = df["sentence"].apply(word2ind)
    df["sentence_len"] = df["sentence"].apply(len)
    print(df.describe())
    x = np.array(df["vector"].tolist())
    y = np.array(df["label"].tolist())
    label_num = len(set(df["label"].tolist()))
    return x, y, label_num


class DealDataset(Dataset):
    def __init__(self, x_train, y_train, device):
        self.x_data = torch.from_numpy(x_train).long().to(device)
        self.y_data = torch.from_numpy(y_train).long().to(device)
        self.len = x_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def evaluate(model, dataloader_dev):
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for datas, labels in dataloader_dev:
            output = model(datas)
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
    model_name = 'text_cnn'
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
        x_train, y_train, label_num = get_data(train_path)
        dataset = DealDataset(x_train, y_train, device)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        x_dev, y_dev, _ = get_data(dev_path)
        dataset_dev = DealDataset(x_dev, y_dev, device)
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        model.train()
        best_acc = 0
        for epoch in range(epochs):
            index = 0
            acc_loss = 0
            train_acc_val = 0
            for datas, labels in tqdm(dataloader):
                model.zero_grad()
                output = model(datas)
                # output (batch_size,label_num_probit)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                index += 1
                acc_loss += loss
                # if index % 50 == 0:
                #     每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                # dev_acc = evaluate(model, dataloader_dev)
                # print(f'epoch:{i} batch:{index} loss:{loss} train_acc:{train_acc} dev_acc:{dev_acc}')
                # if dev_acc > best_acc:
                #     torch.save(model, f'{output_path}/{model_name}/model.pt')
                model.train()
            writer.add_scalar("Loss/train", acc_loss, epoch)
            # tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
            print(f"acc_loss is {acc_loss}")
        torch.save(model, f'{output_path}/{model_name}/model.pt')
        dev_acc = evaluate(model, dataloader_dev)
        print(dev_acc)
        print('train finish')


if __name__ == "__main__":
    main(label_num)
    # get_data(train_path)
