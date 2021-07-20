from typing import List, Tuple, Dict

# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://www.cnblogs.com/hapjin/p/9033471.html
# https://www.cnblogs.com/pinard/p/6945257.html
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq: List[str], to_ix: Dict) -> torch.Tensor:
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_boardcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_boardcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self) -> List[torch.Tensor]:
        return [torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2)]

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        # seq_len
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        #  seq_len,1,embedding_dim
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # (seq_len, batch, num_directions * hidden_size) -> (11, 1, 2 * 2)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # (seq_len, num_directions * hidden_size)
        lstm_features = self.hidden2tag(lstm_out)
        # (seq_len, tagset_size)
        return lstm_features

    def _viterbi_decode(self, feats):
        """
        .. math::
        v_t(j) = max_{i=1}^{N} v_{t-1}(i) a_{ij} b_j(o_t)
        """
        #  TODO
        #  维特比算法的状态转移方程
        # v_t(j)=max_{i=1}^{N}v_{t-1}(i)a_{ij}b_j(o_t)
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def forward(self, sentence: torch.Tensor):
        # bilstm: seq_length
        lstm_fetas = self._get_lstm_features(sentence)
        # crf
        score, tag_seq = self._viterbi_decode(lstm_fetas)
        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data: List[Tuple[List[str], List[str]]] = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #  输入的是tensor 没有batch_size [0,1,2,3,4,5,6]
    print(model(precheck_sent))
