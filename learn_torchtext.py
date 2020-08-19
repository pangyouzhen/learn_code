import spacy
import torch
from torchtext import data, datasets
from torchtext import Vectors
from torch.nn import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spacy_en = spacy.load("en_core_web_sm")


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)

train, val = data.TabularDataset.splits(
    path='.',
    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]
    #     fields None代表不想要的数据
)
