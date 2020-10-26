1. 一个epoch是对sample的一次训练
2. batch 就是将sample绩效宁分割，分成batch个，所以神经网络的第一个入参的维度是batch_size, batch_size 对模型的训练有什么影响，如何进行调节
3. Normalization 归一化和规范化，将是数据转化为 均值为0，方差为1 的正态分布
4. batch normal 和 layer normal 有什么区别
5. attention 的计算公式 attention(Q,K,V)  = softmax(sim (Q,k)) * V
6. 与之对应的是self attention 中的QKV是怎样对应的，
7. 深度学习中解决过拟合的问题的方法有哪些
8. esim的网络结构
9. transformer的网络结构，bert的网络结构，qanet的网络结构，bert的效果优化优化在什么地方
10. qanet的网络结构 
11. qanet embedding：字向量(卷积的字向量)  + 词向量（预训练的词向量）+ highway nets 网络
  12. embedding encoder: encoder blocker -> [(pos-encoding) + conv x # + self-attention + feed-forward]
  13. context-attetion layer：发现context query之间的关系，并在词的层面上，解析出query，context 中的关键词语
  14. model encoder layer: stack blocks
  15. output layer: 阅读理解 -> 在文章中的起始和结束位置
  16. qanet 的网络结构是怎样的？
  17. bert  的网络结构是怎样的？
