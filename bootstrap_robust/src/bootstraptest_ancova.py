import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import norm, chi2, ttest_ind

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

        treatment_code = np.zeros((self.n, np.unique(treatment).shape[0]))
        treatment_code[np.arange(self.n), self.treatment] = 1

        self.design = np.concatenate((np.ones((self.n, 1)),
                                      self.x,
                                      treatment_code[:, 1:]),
                                     axis=1)

        self.design_h0 = np.concatenate((np.ones((self.n, 1)),
                                         self.x),
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

            self.residual = self.y - self.design.dot(self.beta)

            for j in np.unique(self.treatment):
                self.inv_variance[self.treatment == j] = 1 / self.residual[
                    self.treatment == j].var()

            i += 1

        design_weight = self.design * self.inv_variance.reshape((-1, 1))
        self.g = (design_weight).T.dot(self.y - self.design.dot(
            self.beta)) / self.n
        self.d_g = - design_weight.T.dot(self.design) / self.n

        self.v = design_weight.T.dot(self.design) / self.n

        # variance of GMM estimator is
        # \sqrt n (\hat \beta - \beta_0) -> N(0, (G W G^T)^{-1})
        # G = \partial \bar g(beta)^T / \partial \beta

        self.w = np.linalg.inv(self.v)
        self.variance = np.linalg.inv(self.d_g.dot(self.w).dot(
            self.d_g.T)).diagonal() / self.n

        self.gmm = self.g.dot(self.w).dot(self.g) * self.n

        # under the H_0
        i = 0
        while i < self.maxiter:
            beta_h0 = np.linalg.solve(self.design_h0.T.dot(
                np.multiply(self.inv_variance.reshape((-1, 1)), self.design_h0)
            ),
                self.design_h0.T.dot(self.inv_variance *
                                     self.y))

            i += 1

        self.beta_h0 = np.zeros(self.design.shape[1])
        # TODO repalce x.shape[1] + 1 to the number of covariates.
        self.beta_h0[:(self.x.shape[1] + 1)] = beta_h0

        self.g_h0 = design_weight.T.dot(
            self.y - self.design.dot(self.beta_h0)) / \
                    self.n

        self.v_h0 = design_weight * (self.y - self.design.dot(
            self.beta_h0)).reshape((-1, 1))
        self.v_h0 = self.v_h0.T.dot(self.v_h0) / self.n


        self.d_g_h0 = -design_weight.T.dot(self.design) / self.n

        self.gmm_h0 = self.n * self.g_h0.dot(self.w).dot(self.g_h0)

        self.d_gmm_h0 = self.d_g_h0.dot(self.w).dot(self.g_h0)

        self.gwg = self.d_g_h0.dot(self.w.dot(self.v_h0).dot(self.w)).dot(
            self.d_g_h0.T)

        self.score = self.d_gmm_h0.T.dot(np.linalg.inv(
            self.gwg)).dot(self.d_gmm_h0) * self.n

        self.lr = self.gmm_h0 - self.gmm

        logger.info(f"lr={self.lr}, score={self.score}")

    def cuped(self):
        self.theta = np.linalg.solve(self.design_h0[:, [0, 3]].T.dot(
            self.design_h0[:, [0, 3]]),
            self.design_h0[:, [0, 3]].T.dot(self.y))

        self.y_cv = self.y - self.design_h0[:, [0, 3]].dot(self.theta)


if __name__ == '__main__':
    nrep = 5000

    beta_cv = np.zeros((nrep, 1))
    lr = np.zeros(nrep)
    score = np.zeros(nrep)
    variance = np.zeros((nrep, 5))
    n = 400
    np.random.seed(10)
    pv = np.zeros(nrep)
    pv_cv = np.zeros(nrep)

    n_treat = 3
    effect = [0, 0.0, 0.3]
    var = [0.2, 0.2, 0.2]
    beta = np.zeros((nrep, 3 + n_treat))

    variance = np.zeros((nrep, 3 + n_treat))

    for i in range(nrep):
        x = np.random.normal(10, 1, 4 * n).reshape((n, -1))
        treatment = np.random.uniform(0, 2, n).round().astype(int)

        y = np.zeros(n)
        for j in range(3):
            epsilon = np.random.normal(0, var[j], np.sum(treatment == j))
            epsilon = epsilon - epsilon.mean()

            y[treatment == j] = 1 + effect[j] + x[treatment == j,].dot([1, 2, 3,
                                                                        4]) + epsilon

        logger.info(y.var())

        model = BootstrapAncova(x[:, :3], y, treatment, 5)
        model.fit()
        model.cuped()
        y_cv = model.y_cv
        beta[i,] = model.beta
        lr[i] = model.lr
        score[i] = model.score
        variance[i,] = model.variance
        pv[i] = ttest_ind(y[treatment == 0], y[treatment == 2],
                          equal_var=False)[1]
        pv_cv[i] = ttest_ind(y_cv[treatment == 0], y_cv[treatment == 2],
                             equal_var=False)[1]
        beta_cv[i] = y_cv[treatment == 2].mean() - y_cv[treatment == 0].mean()
    plt.figure()
    fig, axs = plt.subplots(5)
    axs[0].hist(beta[:, -1], bins=50, alpha=0.2)
    axs[0].hist(beta_cv, bins=50, alpha=0.5)

    z = (beta[:, 1] - 0) / variance[:, 1] ** 0.5

    axs[1].hist((1 - norm.cdf(abs(z))) * 2, bins=50, alpha=0.5)

    axs[2].hist((1 - chi2.cdf(score, df=2)), bins=50, alpha=0.5)
    axs[3].hist(pv, bins=50, alpha=0.5)
    axs[4].hist(pv_cv, bins=50, alpha=0.5)

    plt.show()
    logger.info(f"pv={np.sum(pv < 0.05) / nrep}, pv_cv="
                f"{np.sum(pv_cv < 0.05) / nrep}, score="
                f"{np.sum(score > chi2.ppf(0.95, df=2)) / nrep}, lr"
                f"={np.sum(lr > chi2.ppf(0.95, df=2)) / nrep}")
