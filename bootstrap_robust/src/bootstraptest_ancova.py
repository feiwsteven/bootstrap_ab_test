"""
Score test for ANCOVA model
"""
import numpy as np
from typing import List
from loguru import logger

__version__ = '0.1'


class BootstrapAncova:
    def __init__(self, x: np.ndarray, y: np.ndarray, treatment: np.ndarray,
                 maxiter: int, parameter_h0: List):
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
        self.parameter_h0 = parameter_h0

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
        self.beta_h0[self.parameter_h0] = beta_h0


        self.g_h0 = design_weight.T.dot(
            self.y - self.design.dot(self.beta_h0)) / \
                    self.n

        self.residual_h0 = (self.y - self.design.dot(
            self.beta_h0)).reshape((-1, 1))
        self.Omega_h0 = (design_weight * self.residual_h0).T.dot(design_weight *
                                                             self.residual_h0) / self.n


        self.d_g_h0 = -design_weight.T.dot(self.design) / self.n

        self.Sigma_h0 = self.d_g_h0.dot(np.linalg.inv(self.Omega_h0)).dot(
            self.d_g_h0.T)

        #self.gmm_h0 = self.n * self.g_h0.dot(self.w).dot(self.g_h0)

        self.gmm_h0 = self.n * self.g_h0.dot(np.linalg.inv(self.Omega_h0)).dot(
            self.g_h0)

        self.d_gmm_h0 = self.d_g_h0.dot(np.linalg.inv(self.Omega_h0)).dot(self.g_h0)


        #self.gwg = self.d_g_h0.dot(self.w.dot(self.v_h0).dot(self.w)).dot(
        #    self.d_g_h0.T)


        #self.score = self.d_gmm_h0.T.dot(np.linalg.inv(
        #    self.gwg)).dot(self.d_gmm_h0) * self.n

        self.score = self.d_gmm_h0.T.dot(np.linalg.inv(
            self.Sigma_h0)).dot(self.d_gmm_h0) * self.n

        self.lr = self.gmm_h0 - self.gmm

        #logger.info(f"lr={self.lr}, score={self.score}")

    def cuped(self, adjusted_covariate: List):
        adjusted_covariate = [0] + adjusted_covariate
        self.theta = np.linalg.solve(self.design_h0[:, adjusted_covariate].T.dot(
            self.design_h0[:, adjusted_covariate]),
            self.design_h0[:, adjusted_covariate].T.dot(self.y))

        self.y_cv = self.y - self.design_h0[:, adjusted_covariate].dot(self.theta)
