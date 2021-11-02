import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

from sklearn.datasets.samples_generator import make_blobs

X: np.ndarray
y: np.ndarray
samples_num = 100
# X, y = make_blobs(n_samples=samples_num, centers=3, random_state=0)
X, y = make_blobs(n_samples=100, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True,
                  random_state=5)
plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="plasma")
plt.show()
print(y)
print(X)
# sklearn 内置方法
from sklearn.cluster import KMeans

Cluster = KMeans(n_clusters=4)
Cluster.fit(X)
y_pred = Cluster.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='plasma')
plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
plt.show()

# from scractchd
rand_num = 4
t = np.random.choice(samples_num, size=rand_num)
# for i in t:
#     print(t[i])
