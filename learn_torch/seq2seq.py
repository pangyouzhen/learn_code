from __future__ import unicode_literals, print_function, division

import random
import re
import unicodedata
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
from torch.utils.data import dataset, dataloader
from torch_seq2seq import normalizeString, Lang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("../data/eng-fra.txt", sep="\t", names=["eng", "fra"])
print(df)
print(df.shape)
df["eng"] = df['eng'].apply(normalizeString)
df["fra"] = df["fra"].apply(normalizeString)
eng_lang = Lang("eng")
fra_lang = Lang("fra")
