import numpy as np


def log_gauss(X, mu, sigma):
    """
    Examples
    --------
    >>> from scipy.stats import multivariate_normal
    >>> mu = np.array([0.0, 0.0])
    >>> sigma = np.array([[3.0, 4.0],
    ...                   [4.0, 10.0]])
    >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> result = log_gauss(X, mu, sigma)
    >>> expected = multivariate_normal.logpdf(X, mu, sigma)
    >>> np.max(np.abs(result - expected)) < 1e-8
    True

    :param X: np.array, shape(n, d)
    :param mu: np.array, shape(d,), giving the mean
    :param sigma: np.array, shape(d, d), giving the covariance matrix
    :returns:
        y: np.array, shape(n,)
    """
    n, d = X.shape
    L = np.linalg.cholesky(sigma)
    X = X - mu
    a = np.sum(np.dot(X, np.linalg.inv(L.T))**2, axis=1)
    b = d * np.log(2*np.pi) + 2 * np.log(np.diag(L)).sum()
    y = -0.5 * (a + b)
    return y


def log_multinormal(X, mu):
    """
    :param X: np.array, shape(n, d)
    :param mu: np.array, shape(d,)
    :returns:
        y: np.array, shape(n,)
    """
    X = X.astype('int64')
    factorial = np.vectorize(lambda x: np.prod(range(1, x+1)))
    log = np.log
    a = log(factorial(X.sum(axis=1)))
    b = np.sum(log(factorial(X)), axis=1)
    c = np.dot(X, log(mu))
    y = a - b + c
    return y
