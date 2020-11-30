from sklearn.datasets import make_blobs

from sklearn.svm import SVC

# 固定随机数种子
samples, target = make_blobs(n_samples=40, n_features=3, centers=2, random_state=0)
print(samples)
print("----------------------------------------------")
svc = SVC(kernel='linear')
svc.fit(samples, target)
# 计算超平面的斜率
w = (-svc.coef_[:, 0] / svc.coef_[:, 1])[0]
# 计算超平面的截距
k = svc.intercept_[0]
print("w=", w)
print("k=", k)
a1 = svc.support_vectors_[0]
a2 = svc.support_vectors_[-1]
print("a1=", a1)
print("a2=", a2)
# TODO 具体计算过程
