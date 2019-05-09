import unittest
import numpy as np
import my_utils as utils

from scipy.stats import multivariate_normal

EPS = 1e-6


class TestMixtureGaussian(unittest.TestCase):
    def test_log_gauss_pdf(self):
        d = 4
        mu = np.random.randn(d)
        s = np.random.randn(40, d)
        sigma = np.dot(s.T, s)
        X = np.random.randn(10, d)
        p1 = np.log(multivariate_normal(mu, sigma).pdf(X))
        p2 = utils.log_gauss_pdf(X, mu, sigma)
        err = np.max(np.abs(p1 - p2))
        self.assertTrue(err < EPS)
