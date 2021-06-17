import numpy as np
from sklearn.metrics import pairwise_distances


def euclidean_distance(x, y):
    """
    :param x: D-dimensional vector
    :param y: D-dimensional vector
    :return: dist - scalar value
    """
    dist = np.sqrt(np.sum((x - y) ** 2))
    return dist


def cost_function(X, K, ind_samples_clusters, centroids):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: cost - a scalar value
    """
    J = 0
    N = X.shape[0]
    J = np.sum(np.sum(ind_samples_clusters * pairwise_distances(X, centroids, metric='euclidean')))

    return J


def closest_centroid(sample, centroids):
    """
    :param sample: a data point x_n (of dimension D)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: idx_closest_cluster - index of the closest cluster
    """
    # Calculate distance of the current sample to each centroid
    # Return the index of the closest centroid (int value from 0 to (K-1))

    distances = pairwise_distances(centroids, sample.reshape(1, -1), metric='euclidean')
    distances = distances ** 2
    idx_closest_cluster = np.argmin(distances)

    return idx_closest_cluster


def assign_samples_to_clusters(X, K, centroids):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    """
    print("test assign_samples_to_clusters")
    N = X.shape[0]  # N - number of samples

    ind_samples_clusters = np.zeros((N, K))
    # print(ind_samples_clusters)
    helper = 0
    for x in X:
        closestC = closest_centroid(x, centroids)
        ind_samples_clusters[helper, closestC] = 1
        helper += 1

    # TODO: implement
    # There will be a function call to closest_centroid function
    # print(ind_samples_clusters)
    assert np.min(ind_samples_clusters) == 0 and np.max(ind_samples_clusters == 1), "These must be one-hot vectors"
    return ind_samples_clusters


def recompute_centroids(X, K, ind_samples_clusters):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    :return: centroids - means of clusters, shape: (K, D)
    """
    D = X.shape[1]

    centroids = np.zeros((K, D))

    print("recompute C")

    centroids = np.dot(X.transpose(), ind_samples_clusters).transpose()

    helper = 0
    for column in ind_samples_clusters.transpose():
        centroids[helper] = centroids[helper] / np.sum(column)

        helper += 1

    print(centroids.shape)
    print(centroids)

    return centroids


def kmeans(X, K, max_iter):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param max_iter:
    :return: ind_samples_clusters - indicator variables for all data points, shape: (N, K)
            centroids - means of clusters, shape: (K, D)
            cost - an array with values of cost over iteration
    """

    n, d = X.shape

    # Init centroids
    rnd_points = np.random.randint(low=0, high=n, size=K)
    centroids = X[rnd_points, :]
    eps = 1e-6

    print(f'Init centroids: {centroids}')

    cost = []
    for it in range(max_iter):
        # Assign samples to the clusters
        ind_samples_clusters = assign_samples_to_clusters(X, K, centroids)  # TODO: function call
        J = cost_function(X, K, ind_samples_clusters, centroids)  # TODO: function call to evaluate cost
        cost.append(J)

        # Calculate new centroids from the clusters
        centroids = recompute_centroids(X, K, ind_samples_clusters)  # TODO: function call
        J = cost_function(X, K, ind_samples_clusters, centroids)  # TODO: function call to evaluate cost again
        cost.append(J)

        if it > 0 and np.abs(cost[-1] - cost[-2]) < eps:
            print(f'Iteration {it + 1}. Algorithm converged.')
            print(f'New centroids: {centroids}')
            break

    return ind_samples_clusters, centroids, cost
