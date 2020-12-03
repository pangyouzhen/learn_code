import numpy as np
from scipy.cluster.vq import vq, kmeans2
from scipy.spatial.distance import cdist


def train(vec, M):
    Ds = int(vec.shape[1] / M)
    codeword = np.empty((M, 256, Ds), np.float32)
    for m in range(M):
        vac_sub = vec[:, m * Ds:(m + 1) * Ds]
        # 将vec 切分成M组，对于每一组进行 k means 聚类
        codeword[m], label = kmeans2(vac_sub, 256)
    return codeword


def encode_(codeword, vec):
    M, _K, Ds = codeword.shape
    pqcode = np.empty((vec.shape[0], M), np.uint8)
    for m in range(M):
        vec_sub = vec[:, m * Ds: (m + 1) * Ds]
        pqcode[:, m], dist = vq(vec_sub, codeword[m])
    return pqcode


def search(codeword, pqcode, query):
    M, _K, Ds = codeword.shape
    dist_table = np.empty((M, 256), np.float32)
    for m in range(M):
        query_sub = query[m * Ds:(m + 1) * Ds]
        # 将query 切分成M组，每一组都和codeword的对应组计算距离
        # print(query_sub.shape)
        dist_table[m, :] = cdist([query_sub],
                                 codeword[m], 'sqeuclidean')[0]
    dist = np.sum(dist_table[range(M), pqcode], axis=1)
    return dist


if __name__ == '__main__':
    M = 3
    query = np.random.randn(1, 150)
    q = query.tolist()[0]
    print(len(q))
    # query = np.array([0.34, 0.22, 0.68, 1.02, 0.03, 0.71])
    vec = np.random.randn(1200, 150)
    codeword = train(vec, M)
    print("总共分为M, _K, Ds", codeword.shape)
    pqcode = encode_(codeword, vec)
    res = (search(codeword, pqcode, q))
    print(res)
