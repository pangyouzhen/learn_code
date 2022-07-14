from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List

import torch
from torch import Tensor
from transformers.modeling_outputs import SequenceClassifierOutput 
from transformers.models.bert import BertModel, BertTokenizer
from transformers import BatchEncoding

tokenizer = AutoTokenizer.from_pretrained("/content/bert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained("/content/bert-base-uncased")


input_text = "今天天气很好啊,你好吗"
# todo input_text2 怎样一起输入
input_text2 = "文本"
# 通过tokenizer把文本变成 token_id
batch_encoding: BatchEncoding = tokenizer(input_text)
print(batch_encoding)
input_ids: List[int] = tokenizer.encode(input_text, add_special_tokens=True)

print(len(input_ids))
print(input_ids)
# 101 代表cls 符号，102代表 sep
# [101, 791, 1921, 1921, 3698, 2523, 1962, 1557, 117, 872, 1962, 1408, 102]
input_ids: Tensor = torch.tensor([input_ids])
# 获得BERT模型最后一个隐层结果
print(input_ids.shape)
with torch.no_grad():
    # bert 的输入为 batch_size,seq_length
    a: SequenceClassifierOutput = model(input_ids)
    # print(a)
    #注意这里的输出和AutoModel的输出类型是不行的
    print(a.logits)