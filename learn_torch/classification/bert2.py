from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

model_name = "/home/pang/Downloads/bert_base_chinese/"
# max sequence length for each document/sentence sample
max_length = 32

tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

train_path = '/data/project/nlp_summary/data/THUCNews/data/train.txt'
dev_path = '/data/project/nlp_summary/data/THUCNews/data/dev.txt'
vocab_path = '/data/project/nlp_summary/data/THUCNews/data/vocab.txt'

train_df = pd.read_csv(train_path, sep="\t", names=["sentence", "label"])
train_texts = train_df["sentence"].tolist()
train_labels = train_df["label"].tolist()
target_names = len(set(train_labels))

valid_df = pd.read_csv(dev_path, sep="\t", names=["sentence", "label"])
valid_texts = valid_df["sentence"].tolist()
valid_labels = valid_df["label"].tolist()

train_encodings: BatchEncoding = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings: BatchEncoding = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)


class DealDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings: BatchEncoding = encodings
        self.labels: List[int] = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = DealDataset(train_encodings, train_labels)
dev_dataset = DealDataset(valid_encodings, valid_labels)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=target_names)

from sklearn.metrics import accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=20,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,  # log & save weights each logging_steps
    evaluation_strategy=IntervalStrategy.STEPS,  # evaluate each `logging_steps`
    # no_cuda=True,
)

trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=dev_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # the callback that computes metrics of interest
)

trainer.train()
trainer.evaluate()
