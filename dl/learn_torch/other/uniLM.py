import torch
import torch.nn as nn

from transformers.models.bert import modeling_bert
from transformers.models.bert.modeling_bert import BertEncoder, BertPreTrainedModel, BertEmbeddings, \
    BertPredictionHeadTransform, BertPooler
from transformers.models.bert import BertModel, BertTokenizer


class BertPreTrainingHeads(nn.Module):
    def __init__(self, bert_model_embedding_weights, config):
        super(BertPreTrainingHeads, self).__init__()
        self.transform = BertPredictionHeadTransform(config)  #
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  #
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BojoneModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BojoneModel, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertPreTrainingHeads(self.embeddings.word_embeddings.weight, config)
        self.init_weights()

    def compute_attention_bias(self, segment_ids):
        idxs = torch.cumsum(segment_ids, dim=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = mask.to(torch.float).unsqueeze(1)
        mask = -(1.0 - mask) * 1e12
        return mask

    def forward(self, input_ids, token_type_ids):
        extended_attention_mask = self.compute_attention_bias(token_type_ids)

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        padding_mask = (input_ids != 0).to(input_ids.device)

        encoded_layers = self.encoder(embedding_output, attention_mask=extended_attention_mask,
                                      head_mask=padding_mask.unsqueeze(0).expand(self.config.num_attention_heads,
                                                                                 token_type_ids.size(0),
                                                                                 token_type_ids.size(1)))
        sequence_output = encoded_layers[-1]

        prediction_scores = self.cls(sequence_output)  # [b,s,V]

        return prediction_scores


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def compute_attention_bias(self, segment_ids):
        idxs = torch.cumsum(segment_ids, dim=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = mask.to(torch.float).unsqueeze(1)
        mask = -(1.0 - mask) * 1e12
        return mask

    def forward(self, input_ids, token_type_ids):
        extended_attention_mask = self.compute_attention_bias(token_type_ids)

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        padding_mask = (input_ids != 0).to(input_ids.device)

        encoded_layers = self.encoder(embedding_output, attention_mask=extended_attention_mask,
                                      head_mask=padding_mask.unsqueeze(0).expand(self.config.num_attention_heads,
                                                                                 token_type_ids.size(0),
                                                                                 token_type_ids.size(1)))
        sequence_output = encoded_layers[-1]

        pooled_output = self.pooler(sequence_output)

        return sequence_output, pooled_output


class BojoneModelWithPooler(BertPreTrainedModel):
    def __init__(self, config):
        super(BojoneModelWithPooler, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = modeling_bert.BertPreTrainingHeads(config)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, output_type="pooler"):
        """
        :param input_ids:
        :param token_type_ids:
        :param output_type:  "seq2seq" or "pooler"
        :return:
        """
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids)

        if output_type == "pooler":
            return pooled_output
        else:
            prediction_scores, _ = self.cls(sequence_output, pooled_output)  # [b,s,V]
            return prediction_scores


if __name__ == '__main__':
    bert_wwm_pt_path = "/data/project/learn_code/data/chinese-bert-wwm-ext"
    config = bert_wwm_pt_path + "/config.json"
    tmp_state_dict = torch.load(bert_wwm_pt_path + "/pytorch_model.bin", map_location="cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_wwm_pt_path)
    bjModel = BojoneModel.from_pretrained(pretrained_model_name_or_path=bert_wwm_pt_path, config=config,
                                          state_dict=tmp_state_dict, local_files_only=False)
