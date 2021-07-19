import torch
from transformers.models.bert import BertModel, BertTokenizer

model_name = '/data/project/learn_code/data/chinese-bert-wwm-ext/'
# 读取模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
# 载入模型
model = BertModel.from_pretrained(model_name)
# 输入文本
# input_text = "Here is some text to encode"
input_text = "今天天气很好啊,你好吗"
# 通过tokenizer把文本变成 token_id
input_ids = tokenizer.encode(input_text, add_special_tokens=True)
print(len(input_ids))
# input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
input_ids = torch.tensor([input_ids])
# 获得BERT模型最后一个隐层结果
print(input_ids.shape)
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]
    print(last_hidden_states)
    print(last_hidden_states.shape)
