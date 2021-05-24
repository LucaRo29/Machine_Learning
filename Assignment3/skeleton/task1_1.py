import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier


def eucledian_dist(p1, p2):
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist


class KNearestNeighborsClassifier(BaseEstimator):

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self._X = X
        self._y = y

        return

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def predict(self, X):
        # TODO: assign class labels
        print('predict')
        print(X.shape)
        print(self._X.shape)

        print(X[0])
        print("_________")
        print(eucledian_dist(self._X[0], X[0]))
        print(eucledian_dist(self._X[0], X[1]))
        print('!!!!!!!')
        print(eucledian_dist(self._X[1], X[0]))
        print(eucledian_dist(self._X[1], X[1]))

        dist = pairwise_distances(self._X, X, metric='euclidean')
        print(dist.shape)
        print(dist)

        print("_________")

        # useful numpy methods: np.argsort, np.unique, np.argmax, np.count_nonzero
        # pay close attention to the `axis` parameter of these methods
        # broadcasting is really useful for this task!
        # See https://numpy.org/doc/stable/user/basics.broadcasting.html
        return
