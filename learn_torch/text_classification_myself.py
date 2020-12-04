import logging
from typing import Callable

import pandas as pd
import torch
from torch.nn import init
from torchtext import data
from torchtext.datasets import TextClassificationDataset
from torchtext.datasets.text_classification import _csv_iterator, _create_data_from_iterator
from torchtext.vocab import Vectors, build_vocab_from_iterator

from learn_torch.DataFrameDataSet import DataFrameDataset


def init_text_match_data(df: pd.DataFrame, tokenizer: Callable, batch_size: int, vectors_name: str, vectors_path: str,
                         device=torch.device("cpu")):
    LABEL = data.Field(sequential=False, use_vocab=False)
    SENTENCE = data.Field(sequential=True, tokenize=tokenizer, lower=True)
    train = DataFrameDataset(df, fields={'sentence1': SENTENCE, 'sentence2': SENTENCE, "label": LABEL})
    # 使用本地词向量
    # torchtext.Vectors 会自动识别 headers
    vectors = Vectors(name=vectors_name, cache=vectors_path)
    # 获取词向量的维度
    vectors_dim = vectors.dim
    # 获取分类的维度
    num_class = len(set([i.label for i in train.examples]))
    print("词向量的维度是", vectors_dim, "分类的维度是", num_class)
    SENTENCE.build_vocab(train, vectors=vectors)  # , max_size=30000)
    # 这里SENTENCE 根据vectors 构建了自己的词向量 ->> SENTENCE.vocab.vectors
    # 如果原始词向量中没有这个词-> 构建成一个0 tensor
    # 当 corpus 中有的 token 在 vectors 中不存在时 的初始化方式.
    SENTENCE.vocab.vectors.unk_init = init.xavier_uniform
    train_iter = data.BucketIterator(train, batch_size=batch_size,
                                     shuffle=True, device=device)
    # train_iter = data.BucketIterator(train, batch_size=batch_size, device=DEVICE)
    return SENTENCE, train_iter, num_class, train


def parser_df(x: str):
    df = pd.read_csv(x, sep=",")
    print(df.shape)


def _setup_datasets(dataset_name, root='.data', ngrams=1, vocab=None, include_unk=False, downloads=False, path=None):
    logging.info('Building Vocab based on {}'.format(path))
    vocab = build_vocab_from_iterator(_csv_iterator(path, ngrams))
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(path, ngrams, yield_cls=True), include_unk)

    return TextClassificationDataset(vocab, train_data, train_labels)


def AG_NEWS(*args, **kwargs):
    return _setup_datasets(*(("AG_NEWS",) + args), **kwargs)


if __name__ == '__main__':
    parser_df("./.data/ag_news_csv/train.csv")
