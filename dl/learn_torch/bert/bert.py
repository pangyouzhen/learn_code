from typing import List

import torch
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert import BertModel, BertTokenizer
from transformers import BatchEncoding
from torch.nn.modules.transformer import MultiHeadAttention

model_name = '/data/project/learn_code/data/chinese-bert-wwm-ext/'
# 读取模型对应的tokenizer
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
# 载入模型
model: BertModel = BertModel.from_pretrained(model_name)
# 输入文本
# input_text = "Here is some text to encode"
input_text = ["今天天气很好啊,你好吗",'you know']

encode_input :BatchEncoding = tokenizer.batch_encode_plus(
    input_text,
    add_special_tokens = True,
    padding = True,
    max_length = 256,
    pad_max_length = True,
    return_attention_mask= True,
)
# 101 代表cls 符号，102代表 sep
# [101, 791, 1921, 1921, 3698, 2523, 1962, 1557, 117, 872, 1962, 1408, 102]
# 获得BERT模型最后一个隐层结果
for i in encode_input.keys():
    encode_input[i] = torch.tensor(encode_input[i])
with torch.no_grad():
    # bert 的输入为 batch_size,seq_length
    a: BaseModelOutputWithPoolingAndCrossAttentions = model(**encode_input)
    last_hidden_state = a.last_hidden_state
    # batch_size,seq_length,embedding
    print(torch.equal(last_hidden_state[:, 0, :], a.pooler_output))
    # last_hidden_state 的输出为 batch_size,seq_length, embedding
    print(last_hidden_state)
    print(last_hidden_state.shape)
