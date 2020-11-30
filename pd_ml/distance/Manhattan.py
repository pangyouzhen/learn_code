import numpy as np

from pd_ml.distance.Distance import Distance


class Manhattan(Distance):
    def distance(self, x, y):
        res = (sum([abs(i - j) for i, j in zip(x, y)]))
        return res


if __name__ == '__main__':
    x = np.random.random(10)
    y = np.random.random(10)
    man = Manhattan()
    man_distance = man.distance(x, y)
    print(man_distance)
