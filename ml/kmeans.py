import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import faiss

# faiss 安装的命令是 faiss-cpu 和 faiss-gpu

print(sk.__version__)
np.random.seed(0)

from sklearn.datasets import make_blobs

X: np.ndarray
y: np.ndarray
samples_num = 100
X, y = make_blobs(n_samples=100, n_features=4, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True,
                  random_state=5)
# plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="plasma")
# plt.show()
# print(y)
# print(X)
# sklearn 内置方法
from sklearn.cluster import KMeans

Cluster = KMeans(n_clusters=3)
Cluster.fit(X)
y_pred = Cluster.predict(X)
# print(Cluster.cluster_centers_)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='plasma')
# plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
# plt.show()

#  使用Faiss进行聚类
print(type(X))
X = X.astype("float32")
d = 4
index = faiss.IndexFlatL2(d)
index.add(X)
# k = 4                          # we want 4 similar vectors
# D, I = index.search(xq, k)
ncentroids = 3
niter = 20
verbose = True
d = X.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
kmeans.train(X)
print(kmeans.centroids)
