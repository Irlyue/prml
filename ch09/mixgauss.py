import numpy as np
import my_utils as mu


class MixtureGaussian:
    def __init__(self, n_components, init='random', n_iters=300, seed=None,
                 print_every=None):
        """
        Arguments
        ---------
        :param n_components: int, number of gaussian components
        :param init: str, method to initialize the responsibility matrix.
            "random": randomly assign each sample to one of the component
        :param n_iters: int, number of iterations
        :param seed: int, setting it with the same seed will gaurantee
        the same result when run several times
        :param print_every: int, frequency to print logging information
        """
        assert init in ('random',)
        wanted = locals()
        wanted.pop('self')
        self.save_context(**wanted)
        # freeze it for representation
        self.freezed = vars(self).copy()

    def fit(self, X):
        self.on_start_fit(X)
        for i in range(self.n_iters):
            self.step(i)
            if self.print_every and i % self.print_every == 0:
                print('{:<4}: {:.2f}'.format(i, self.llh))

    def save_context(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def on_start_fit(self, X):
        if self.seed:
            np.random.seed(self.seed)
        # initialization
        if self.init == 'random':
            m = X.shape[0]
            label = np.random.randint(self.n_components, size=m)
            R = np.zeros((m, self.n_components))
            R[range(m), label] = 1.
        self.save_context(X=X, R=R)

    def step(self, i):
        self.remove_empty_clusters()
        self.maximization()
        self.expectation()

    def remove_empty_clusters(self):
        R = self.R
        label = np.argmax(R, axis=1)
        R = R[:, np.unique(label)]
        self.save_context(R=R)

    def expectation(self):
        X = self.X
        pis, mus, sigmas = self.pis, self.mus, self.sigmas
        m = X.shape[0]
        R = np.zeros((m, self.n_components))
        for i in range(self.n_components):
            R[:, i] = mu.log_gauss_pdf(X, mus[i], sigmas[i])

        R = np.exp(R) * pis
        llh = np.log(np.sum(R, axis=1)).sum()
        R /= np.sum(R, axis=1, keepdims=True)
        self.save_context(R=R, llh=llh)

    def maximization(self):
        X, R = self.X, self.R
        m, d = X.shape
        nk = np.sum(R, axis=0)
        pis = nk / m
        mus = np.dot(R.T, X) / nk.reshape((-1, 1))
        sigmas = np.zeros((self.n_components, d, d))
        for i in range(self.n_components):
            t = self.X - mus[i]
            rt = t * R[:, i].reshape((-1, 1))
            sigmas[i] = np.dot(rt.T, t) / nk[i] + np.eye(d) * 1e-6
        self.save_context(pis=pis, mus=mus, sigmas=sigmas)

    def __repr__(self):
        properties = ', '.join('{}={}'.format(key, value) for key, value in self.freezed.items())
        return '{}({})'.format(type(self).__name__, properties)
