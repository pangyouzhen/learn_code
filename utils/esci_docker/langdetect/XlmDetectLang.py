from typing import List

import torch
from torch import device
from torch.utils.data import DataLoader, TensorDataset
from transformers import (XLMRobertaForSequenceClassification,
                          XLMRobertaTokenizer)
from .DetectLang import DetectLang

class XlmDetectLang(DetectLang):
    def __init__(self, model_path=None,**kwargs) -> None:
        super().__init__(model_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.module_path)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(self.module_path)
        if "device" not in kwargs.keys():
            self.device = torch.device("cpu")
        self.model.to(self.device)
        # https://huggingface.co/papluca/xlm-roberta-base-language-detection
        # text = ["我今天","I LOVE me"]

    def detect_lang(self,text:List[str],batch_size=None) -> List:
        encode_input = self.tokenizer.batch_encode_plus(
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

        if not batch_size:
            batch_size = 256

        dataset_loader = DataLoader(
            dataset_use,
            batch_size=batch_size
        )


        prediction_class_id = []
        for index,batch in enumerate(dataset_loader):
            batch = tuple(b.to(self.device) for b in batch)
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
            }
            with torch.no_grad():
                logits = self.model(**inputs).logits
                prediction_class_id.extend(list(logits.argmax(-1).cpu()))
        return [self.model.config.id2label[i.item()] for i in prediction_class_id]
