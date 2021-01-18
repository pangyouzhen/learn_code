def attention(query, key, value, mask=None, dropout=None):
    # query size : (batch,注意力头的个数,nums_seq, ??) 最后一个维度不清楚是啥
    d_k = query.size(-1)


