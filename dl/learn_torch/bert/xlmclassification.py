from torch import device
import torch
from transformers import XLMRobertaTokenizer,XLMRobertaForSequenceClassification
from torch.utils.data import TensorDataset,DataLoader
device = torch.device("cpu")

tokenizer = XLMRobertaTokenizer.from_pretrained("")
model = XLMRobertaForSequenceClassification.from_pretrained("")

model.to(device)
# https://huggingface.co/papluca/xlm-roberta-base-language-detection
text = ["我今天","I LOVE me"]

encode_input = tokenizer.batch_encode_plus(
    text,
    add_special_tokens = True,
    padding = True,
    max_length = 256,
    pad_max_length = True,
    return_attention_mask= True,
)

input_ids = torch.tensor(encode_input["input_ids"])
attention_mask = torch.tensor(encode_input["attention_mask"])

dataset_use = TensorDataset(input_ids,attention_mask)

batch_size = 10

dataset_loader = DataLoader(
    dataset_use,
    batch_size=batch_size
)


prediction_class_id = None
for index,batch in enumerate(dataset_loader):
    batch = tuple(b.to(device) for b in batch)
    inputs = {
        "input_ids":batch[0],
        "attention_mask":batch[1],
    }
    with torch.no_grad():
        logits = model(**input).logits
        prediction_class_id = logits.argmax(-1)
for i in prediction_class_id:
    print(model.config.id2label[i.item()])