import numpy as np


def cross_distance(X, Y):
    """
    Arguments
    ---------
    :param X: np.array, shape (m, d)
    :param Y: np.array, shape (n, d)

    Returns
    -------
    :return: np.array, shape (m, n), giving the distance bewteen every two
    points.
    """
    m, n = X.shape[0], Y.shape[0]
    return np.sum(X**2, axis=1).reshape((m, 1)) + \
        np.sum(Y**2, axis=1).reshape((1, n)) - 2*np.dot(X, Y.T)


def index2one_hot(y, k, normalize=False):
    """
    [0, 1, 0, 2] --> [[1, 0, 0],
                      [0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 1]]

    Arguments
    ---------
    :param y: np.array, shape (m,), the class index of each sample
    :param k: int, number of classes
    :param normalize: bool, whether to normalize the one hot matrix
    """
    m = y.size
    oh = np.zeros((m, k))
    oh[range(m), y] = 1.
    if normalize:
        oh /= oh.sum(axis=0, keepdims=True)
    return oh


class KMeans:
    def __init__(self, n_clusters, seed=None, n_steps=None):
        self.n_clusters = n_clusters
        self.seed = seed
        self.n_steps = int(1e7) if n_steps is None else n_steps

    def fit(self, X):
        """
        Arguments
        ---------
        :param X: np.array, shape (m, d)

        Returns
        -------
        :return: tuple of length 2 giving the label and data centers.
        """
        if self.seed:
            np.random.seed(self.seed)
        m, d = X.shape
        # initialization
        mu = X[np.random.permutation(m)[:self.n_clusters]]
        label = cross_distance(X, mu).argmin(axis=1)

        last = np.zeros_like(label)
        for step in range(self.n_steps):
            if np.all(label == last):
                break
            # remove empty clusters
            _, last = np.unique(label, return_inverse=True)
            # the new centers are linear combination of the samples
            w = index2one_hot(label, self.n_clusters, True)
            mu = np.dot(w.T, X)
            label = cross_distance(X, mu).argmin(axis=1)
        return label, mu
