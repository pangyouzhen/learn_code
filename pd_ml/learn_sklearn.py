#  决策树
from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
# 如何进行特征选择 h(x) = -sum(p*log(p))
## 特征选择的两种方法
# 1. 信息增益
# 2. 信息增益比，信息增益没有考虑本身的熵
# 根据两种不同的特征选择方法，存在两种不同的算法ID3，C4
# 如何避免过拟合

# GBDT
from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier()
gbdt.fit(X,Y)
