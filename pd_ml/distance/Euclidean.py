import math

import numpy as np

from pd_ml.distance.Distance import Distance


class Euclidean(Distance):

    def distance(self, x, y):
        assert len(x) == len(y)
        res = math.sqrt((sum([(i - j) ** 2 for i, j in zip(x, y)])))
        return res


if __name__ == '__main__':
    x = np.array([0.53513515, 0.70174702, 0.27492757, 0.32986405, 0.53072557,
                  0.23669886, 0.09919752, 0.62595752, 0.18946333, 0.27596878])
    y = np.array([0.23086819, 0.7189609, 0.14626943, 0.6930221, 0.20841688,
                  0.39647644, 0.64947037, 0.04300404, 0.59875791, 0.98409463])
    eucildean = Euclidean()
    print(eucildean.distance(x, y))
