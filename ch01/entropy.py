import math

from collections import Counter


def entropy(x):
    """
    Examples
    --------
    >>> x = list(range(10))
    >>> math.fabs(entropy(x) - math.log2(10)) < 1e-8
    True

    :param x:
    :return:
    """
    c = Counter(x)
    n = len(x)
    probs = [float(value) / n for key, value in c.items()]
    h = -sum(p * math.log2(p) for p in probs)
    h = max(0.0, h)  # in case x = [1.0], h=-0.0
    return h


def joint_entropy(x, y):
    """
    Examples
    --------
    >>> x = [0, 0, 1, 1]
    >>> y = [0, 1, 0, 1]
    >>> math.fabs(joint_entropy(x, y) - math.log2(4)) < 1e-8
    True
    >>> x = [0, 0, 0, 0]
    >>> y = [0, 0, 0, 1]
    >>> math.fabs(joint_entropy(x, y) - entropy(y)) < 1e-8
    True

    :param x:
    :param y:
    :return:
    """
    assert len(x) == len(y), "`x` and `y` should have the same number of elements!"
    xy = [(i, j) for i, j in zip(x, y)]
    c = Counter(xy)
    n = len(xy)
    probs = [float(value) / n for key, value in c.items()]
    h = -sum(p*math.log2(p) for p in probs)
    h = max(0.0, h)
    return h


def cond_entropy(x, y):
    """
    Conditional entropy H[x|y].
    :param x:
    :param y:
    :return:
    """
    h = joint_entropy(x, y) - entropy(y)
    h = max(0.0, h)
    return h


def mutual_info(x, y):
    """
    Mutual information I[x, y], using the formula:
        I[x, y] = H[x] + H[y] - H[x, y] = H[x] - H[x|y] = H[y] - H[y|x]

    Consider `y` to be the cluster label, then mutual information `H[x] - H[x|y]`
    tells us the reduction in the entropy of class labels that we get if we know
    the cluster labels. (Similar to Information gain in decision trees).
    :param x:
    :param y:
    :return:
    """
    hx = entropy(x)
    hy = entropy(y)
    hxy = joint_entropy(x, y)
    ixy = hx + hy - hxy
    ixy = max(0.0, ixy)
    return ixy


def normalized_mutual_info(x, y):
    """
    Normalized mutual information(NMI):
        NMI = I[x, y] / sqrt(H[x]*H[y])

    Examples
    --------
    >>> x = [0, 0, 1, 1]
    >>> y = [1, 1, 0, 0]
    >>> normalized_mutual_info(x, y)
    1.0
    >>> normalized_mutual_info(x, x)
    1.0
    >>> x = [0, 0, 0, 0]
    >>> y = [0, 1, 2, 3]
    >>> normalized_mutual_info(x, y)
    0.0
    """
    ixy = mutual_info(x, y)
    hx = entropy(x)
    hy = entropy(y)
    nmi = ixy / max(math.sqrt(hx * hy), 1e-10)
    return nmi


def relative_entropy(x, y):
    """
    Relative entropy, namely the KL divergence:
        KL(p(x)||p(y))
    Examples
    --------
    >>> x = [0, 0, 1, 1]
    >>> relative_entropy(x, x)
    0.0
    >>> x = [0, 0, 0, 1, 2, 2]
    >>> y = [0, 1, 1, 2, 2, 2]
    >>> math.fabs(relative_entropy(x, y) - 0.4308) < 1e-4
    True
    """
    cx = Counter(x)
    cy = Counter(y)
    assert cx.keys() == cy.keys(), "`x` and `y` have different value set, current implementation not supported!"
    px = [float(value) / len(x) for key, value in cx.items()]
    py = [float(value) / len(y) for key, value in cy.items()]
    kl = -sum(p*math.log2(q / p) for p, q in zip(px, py))
    kl = max(0.0, kl)
    return kl

