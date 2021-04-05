import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import metrics
from numpy.random import default_rng

np.random.seed(0)

from sklearn.datasets.samples_generator import make_blobs

X: np.ndarray
y: np.ndarray
X, y = make_blobs(n_samples=100, centers=5, random_state=5)
plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
plt.scatter(X[:, 0], X[:, 1])
# plt.show()
print(y)
print(X)
# from sklearn.cluster import KMeans
#
# Cluster = KMeans(n_clusters=5)
# Cluster.fit(X)
# y_pred = Cluster.predict(X)
#
# plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='plasma')
# plt.rcParams.update({'figure.figsize': (10, 7.5), 'figure.dpi': 100})
# plt.show()
rng = default_rng()
rand_X: np.ndarray = rng.choice(20, size=5, replace=False)

print(rand_X)
init_center: np.ndarray = X[rand_X]
