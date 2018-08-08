import numpy as np

from collections import namedtuple


def fit_line(X, t, reg=0.0):
    """
    Fit a line between the fetaure `X` and the target `t`.

    :param X: np.array, shape(n, d)
    :param t: np.array, shape(n,)
    :param reg: float, regularization strength
    :return:
        model: (w, xbar, beta), w[0] is the bias term.
    """
    Model = namedtuple('Model', 'w xbar beta')
    n, d = X.shape
    xbar = X.mean(axis=0, keepdims=True)
    tbar = t.mean()
    X = X - xbar
    t = t - tbar
    X = np.c_[np.ones(n), X]
    xtx = np.dot(X.T, X)
    xtx[range(d), range(d)] += reg
    w = np.linalg.inv(xtx).dot(X.T).dot(t)
    beta = 1.0 / np.var(np.dot(X, w) - t)
    w[0] += tbar - np.dot(w[1:], xbar[0])
    return Model(w, xbar, beta)


def predict(X, model):
    """
    :param X: np.array, shape(n, d)
    :param model: namedtuple(w, xbar, beta)
    """
    n, d = X.shape
    X = np.c_[np.ones(n), X]
    t = np.dot(X, model.w)
    return t
