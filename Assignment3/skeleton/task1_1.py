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

        if (self.k >= self._X.shape[0]):
            print("k must be smaller than the number of data samples")
            return

        # print("_________")
        # print(eucledian_dist(self._X[0], X[0]))
        # print(eucledian_dist(self._X[0], X[1]))
        # print('!!!!!!!')
        # print(eucledian_dist(self._X[1], X[0]))
        # print(eucledian_dist(self._X[1], X[1]))

        # print(dist)

        predictions = np.array([])
        y_train = self._y.reshape(-1, 1)
        for x in X:  # iterating through every test data point
            dist = pairwise_distances(self._X, x.reshape(1, -1), metric='euclidean')
            neighbors = np.concatenate((dist, y_train), axis=1)
            neighbors_sorted = neighbors[neighbors[:, 0].argsort()]  # sorts training points on the basis of distance

            k_neighbors = neighbors_sorted[:self.k]  # selects k-nearest neighbors

            labels, occurences = np.unique(k_neighbors[:, -1], return_counts=True)

            predicted_y = labels[np.argmax(occurences)]

            predictions = np.append(predictions, predicted_y)

        return predictions
