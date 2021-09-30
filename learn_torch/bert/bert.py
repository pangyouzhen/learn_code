from typing import List

import torch
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert import BertModel, BertTokenizer
from transformers import BatchEncoding

model_name = '/data/project/learn_code/data/chinese-bert-wwm-ext/'
# 读取模型对应的tokenizer
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
# 载入模型
model: BertModel = BertModel.from_pretrained(model_name)
# 输入文本
# input_text = "Here is some text to encode"
input_text = "今天天气很好啊,你好吗"
# todo input_text2 怎样一起输入
input_text2 = "文本"
# 通过tokenizer把文本变成 token_id
batch_encoding: BatchEncoding = tokenizer(input_text)
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
    a: BaseModelOutputWithPoolingAndCrossAttentions = model(input_ids)
    last_hidden_state = a.last_hidden_state
    # batch_size,seq_length,embedding
    print(torch.equal(last_hidden_state[:, 0, :], a.pooler_output))
    # last_hidden_state 的输出为 batch_size,seq_length, embedding
    print(last_hidden_state)
    print(last_hidden_state.shape)
