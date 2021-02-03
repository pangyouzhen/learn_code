import re

import pandas as pd
import spacy
import torch
import torchtext
from torchtext import data
from torchtext.datasets import SNLI

from learn_torch.torch_esim import init_text_match_data, init_model, training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_text(s):
    # 移除括号
    s = re.sub('\\(', '', s)
    s = re.sub('\\)', '', s)
    # 使用一个空格替换两个以上连续空格
    s = re.sub('\\s{2,}', ' ', s)
    return s.strip()


spacy_en = spacy.load('en_core_web_md')

label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}


def tokenizer(text):  # create a tokenizer function
    """
    定义分词操作
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def prepare_data():
    data_dir = "/data/Downloads/data/dataset/snli_1.0/snli_1.0_train.txt"
    df = pd.read_csv(data_dir, sep="\t")
    df1 = df[["gold_label", "sentence1_parse", "sentence2_parse"]]
    df1["sentence1_parse"] = df1["sentence1_parse"].apply(extract_text)
    df1["sentence2_parse"] = df1["sentence2_parse"].apply(extract_text)
    df1 = df1[df1["gold_label"].apply(lambda x: x in label_set.keys())]
    df1["gold_label"] = df1["gold_label"].map(label_set)
    df1.columns = ["label", "sentence1", "sentence2"]
    return df1


if __name__ == '__main__':
    LABEL = data.Field(sequential=False, use_vocab=False)
    SENTENCE1 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    SENTENCE2 = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    df = prepare_data()
