import kmeans
import unittest
import numpy as np

EPS = 1e-6


class TestKMeans(unittest.TestCase):
    def test_cross_distance(self):
        np.random.seed(2)
        m, n, d = 40, 50, 10
        X = np.random.randn(m, d)
        Y = np.random.randn(n, d)
        expected = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                expected[i, j] = np.sum((X[i] - Y[j])**2)
        target = kmeans.cross_distance(X, Y)
        self.assertTrue(np.max(np.abs(target - expected)) < EPS)
