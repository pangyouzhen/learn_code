from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#  cross entropy
batch_size = 10
num_class = 12
logits= torch.rand(batch_size,num_class)
#  target 可以是teacher的分布logit，也可以直接使用总的结果
target = torch.randint(low=0,high=1,size=(batch_size,))
ce_loss = nn.CrossEntropyLoss()
# 输入的是logit，未进行归一化
# 必须是delta分布？
#  数学上p是真实分布，放在前面，交叉熵和kl散度都是如此，torch使用的时候是先预测的，后真实的
res = ce_loss(logits,target)
print(f"{res = }")

# nll loss

nll_loss = nn.NLLLoss()
nl_loss = nll_loss(torch.log(torch.softmax(logits,dim=-1)),target)
print(f"{nl_loss = }")
#交叉熵= -似然函数


# KL散度-相对熵-常用生成模型
target_logits= torch.rand(batch_size,num_class)
kl_loss_func = nn.KLDivLoss()
kl = kl_loss_func(torch.log(torch.softmax(logits,dim=-1)),torch.softmax(target_logits,dim=-1))
print(f"{kl}")

# CE = IE + KL散度

ce_loss_each_sample = nn.CrossEntropyLoss(reduction="none")
ce_loss_sample = ce_loss_each_sample(logits,torch.softmax(target_logits,dim=-1))
print(f"{ce_loss_sample = }")
kl_loss_func_sample = nn.KLDivLoss(reduction="none")
kl_sample = kl_loss_func_sample(torch.log(torch.softmax(logits,dim=-1)),torch.softmax(target_logits,dim=-1)).sum(-1)
print(f"{kl_sample}")

# IE计算
target_ie=torch.distributions.Categorical(probs = torch.softmax(target_logits,dim=-1)).entropy()
print(f"{target_ie = }")

print(torch.allclose(ce_loss_sample,kl_sample + target_ie))

# BCEloss
## 可以用nllloss来代替bce
bce_loss_fn = nn.BCELoss()
logits = torch.randn(batch_size)
prob_1 = torch.sigmoid(logits)
target_bce = torch.randint(2,size =(batch_size,)).float()
bce_loss = bce_loss_fn(prob_1,target_bce)
print(f"{bce_loss = }")

# cos