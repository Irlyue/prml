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


def index2one_hot(y, k):
    assert y.ndim == 1
    m = y.size
    oh = np.zeros((m, k), dtype=np.int32)
    oh[range(m), y] = 1
    return oh


def log_gauss_pdf(X, mu, sigma):
    """
    Arguments
    ---------
    :param X: np.array, shape (m, d)
    :param mu: np.array, shape (d,)
    :param sigma: np.array, shape (d, d)

    Return
    ------
    :return: np.array with shape (m,)
    """
    d = sigma.shape[0]
    L = np.linalg.cholesky(sigma)
    Q = np.dot(X - mu, np.linalg.inv(L.T))
    q = np.sum(Q**2, axis=1)
    c = d*np.log(2*np.pi) + 2*np.sum(np.log(L.diagonal()))
    return -0.5 * (q + c)
