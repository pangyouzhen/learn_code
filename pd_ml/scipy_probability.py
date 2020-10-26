# 伯努利分布 -> 0,1 分布
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x1 = np.arange(0, 2, 1)
print(x1)

p1 = 0.5
plist = stats.bernoulli.pmf(x1, p1)
print(plist)

plt.plot(x1, plist, marker="o", linestyle="None")
plt.vlines(x1, 0, plist)
plt.xlabel("randomVariable")
plt.ylabel("p")
plt.title("bernoulli distribute")
plt.show()
