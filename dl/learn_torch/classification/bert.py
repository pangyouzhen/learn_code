import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers.trainer_utils import IntervalStrategy
from transformers import BatchEncoding



def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available


set_seed(1)

model_name = "/data/project/learn_code/data/bert-base-uncased/"
# max sequence length for each document/sentence sample
max_length = 512

tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)


def read_20newsgroups(test_size=0.2):
    # download & load 20newsgroups dataset from sklearn's repos
    dataset = fetch_20newsgroups(data_home="/data/project/learn_code/learn_torch/.data",subset="all", shuffle=True, remove=("headers", "footers", "quotes"))
    documents = dataset.data
    labels = dataset.target
    # split into training & testing a return data as well as label names
    return train_test_split(documents, labels, test_size=test_size), dataset.target_names


# call the function
(train_texts, valid_texts, train_labels, valid_labels), target_names = read_20newsgroups()

train_encodings: BatchEncoding = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings: BatchEncoding = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)


class NewsGroupsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings: BatchEncoding = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


# convert our tokenized data into a torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(target_names)).to("cuda")

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
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=20,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
    load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    logging_steps=200,  # log & save weights each logging_steps
    evaluation_strategy=IntervalStrategy.STEPS,  # evaluate each `logging_steps`
    no_cuda=True,
)

trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=valid_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # the callback that computes metrics of interest
)

trainer.train()
trainer.evaluate()


def get_prediction(text):
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    return target_names[probs.argmax()]


text = """
The first thing is first. 
If you purchase a Macbook, you should not encounter performance issues that will prevent you from learning to code efficiently.
However, in the off chance that you have to deal with a slow computer, you will need to make some adjustments. 
Having too many background apps running in the background is one of the most common causes. 
The same can be said about a lack of drive storage. 
For that, it helps if you uninstall xcode and other unnecessary applications, as well as temporary system junk like caches and old backups.
"""
print(get_prediction(text))
