import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import chi2

__version__ = '0.1'


class BootstrapAncova:
    def __init__(self, x: np.ndarray, y: np.ndarray, treatment: np.ndarray,
                 maxiter: int):
        self.x = x
        self.y = y
        self.treatment = treatment
        self.n = y.shape[0]
        self.maxiter = maxiter

        self.pi = self.treatment.sum() / self.n

        self.x_centered = self.x - self.x.mean(axis=0)
        self.y_centered = self.y - self.y.mean(axis=0)

        self.design = np.concatenate((np.ones((self.n, 1)),
                                      self.treatment.reshape((-1, 1)),
                                      self.x_centered),
                                     axis=1)

        self.design_h0 = np.concatenate((np.ones((self.n, 1)),
                                         self.x_centered),
                                        axis=1)

        self.beta = np.zeros(self.design.shape[1])
        self.beta_h0 = np.zeros(self.design.shape[1])
        self.inv_variance = np.ones(self.n)
        self.inv_variance_h0 = np.ones(self.n)
        self.residual = np.ones(self.n)
        self.residual_h0 = np.ones(self.n)

    def fit(self, method='OLS'):
        """
        Fit ANCOVA model
        :param method:
        :return:
        """

        i = 0
        while i < self.maxiter:
            self.beta = np.linalg.solve(self.design.T.dot(
                np.multiply(self.inv_variance.reshape((-1, 1)), self.design)
            ),
                self.design.T.dot(self.inv_variance *
                                  self.y))

            logger.info(self.beta)

            self.residual = self.y - self.design.dot(self.beta)

            self.inv_variance[self.treatment == 0] = 1 / self.residual[
                self.treatment
                == 0].var()

            self.inv_variance[self.treatment == 1] = 1 / self.residual[
                self.treatment
                == 1].var()

            i += 1


        self.g = self.design.T.dot(self.y - self.design.dot(self.beta)) / self.n

        self.sn = self.design[self.treatment==1,].T.dot(self.design[
                                                           self.treatment==1,
                                                        ]) / self. n * \
                  self.residual[self.treatment == 1].var() + self.design[self.treatment==0,].T.dot(self.design[
                                                           self.treatment==0,
                                                        ]) / self. n * \
                  self.residual[self.treatment == 0].var()

        self.xtx = self.design.T.dot(self.design) / self. n
        # variance of GMM estimator is
        # \sqrt n (\hat \beta - \beta_0) -> N(0, (G W G^T)^{-1})
        # G = \partial \bar g(beta)^T / \partial \beta
        self.variance = np.linalg.inv(self.xtx.dot(np.linalg.inv(self.sn)).dot(
            self.xtx)).diagonal() / self. n


        self.gmm = self.g.dot(np.linalg.inv(self.sn)).dot(self.g) * self.n

        i = 0
        while i < self.maxiter:
            self.beta_h0 = np.linalg.solve(self.design_h0.T.dot(
                np.multiply(self.inv_variance.reshape((-1, 1)),
                            self.design_h0)
            ),
                self.design_h0.T.dot(self.inv_variance *
                                     self.y))

            logger.info(self.beta_h0)

            self.residual_h0 = self.y - self.design_h0.dot(self.beta_h0)

            self.inv_variance_h0 = 1 / self.residual_h0.var()
            i += 1

        self.g_h0 = self.design_h0.T.dot(self.y - self.design_h0.dot(
            self.beta_h0)) / self.n

        self.sn_h0 = self.design_h0[self.treatment==1,].T.dot(self.design_h0[
                                                           self.treatment==1,
                                                        ]) / self. n * \
                  self.residual[self.treatment == 1].var() + self.design_h0[
                         self.treatment==0,].T.dot(self.design_h0[
                                                           self.treatment==0,
                                                        ]) / self. n * \
                  self.residual[self.treatment == 0].var()


        self.gmm_h0 = self.n * self.g_h0.dot(np.linalg.inv(
            self.sn_h0)).dot(self.g_h0)

        self.lr = self.gmm_h0 - self.gmm

        logger.info(self.lr)


if __name__ == '__main__':
    nrep = 5000
    beta = np.zeros((nrep, 6))
    lr = np.zeros(nrep)
    variance = np.zeros((nrep, 6))
    n = 400
    for i in range(nrep):
        x = np.random.normal(0, 1, 4 * n).reshape((n, -1))
        treatment = np.random.binomial(1, 0.5, n)
        y = 1 + treatment * 0 + x.dot([1,2,3,4]) + np.random.normal(0, 0.5, n)

        model = BootstrapAncova(x, y, treatment, 5)
        model.fit()
        beta[i,] = model.beta
        lr[i] = model.lr
        variance[i,] = model.variance

    plt.figure()
    plt.hist(beta[:, 1], bins=50)
    plt.axvline(0.1)
    plt.show()

    plt.hist(lr, bins=50)
    plt.show()
