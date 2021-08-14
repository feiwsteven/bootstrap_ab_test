"""
GMM Score test for ANCOVA model
"""
from typing import List

import numpy as np

__version__ = '0.1'


class CovariateAdjustedAncova:
    """
    Class for GMM score test for ANCOVA model
    """
    def __init__(self, x_mat: np.ndarray, y_vec: np.ndarray, treatment:
    np.ndarray,
                 maxiter: int, parameter_h0: List):
        self.x_mat = x_mat
        self.y_vec = y_vec
        self.treatment = treatment
        self.sample_size = y_vec.shape[0]
        self.maxiter = maxiter

        self.pi_parameter = self.treatment.sum() / self.sample_size

        self.x_centered = self.x - self.x.mean(axis=0)
        self.y_centered = self.y_vec - self.y_vec.mean(axis=0)

        treatment_code = np.zeros((self.sample_size, np.unique(treatment).shape[0]))
        treatment_code[np.arange(self.sample_size), self.treatment] = 1

        self.design = np.concatenate((np.ones((self.sample_size, 1)),
                                      self.x,
                                      treatment_code[:, 1:]),
                                     axis=1)

        self.design_h0 = np.concatenate((np.ones((self.sample_size, 1)),
                                         self.x),
                                        axis=1)

        self.beta = np.zeros(self.design.shape[1])
        self.beta_h0 = np.zeros(self.design.shape[1])
        self.inv_variance = np.ones(self.sample_size)
        self.inv_variance_h0 = np.ones(self.sample_size)
        self.residual = np.ones(self.sample_size)
        self.residual_h0 = np.ones(self.sample_size)
        self.parameter_h0 = parameter_h0

        self.y_cv = None

        self.g_score = None
        self.d_g_score = None
        self.v_mat = None
        self.w_mat = None
        self.variance = None
        self.gmm = None

        self.g_score_h0 = None
        self.gmm_h0 = None
        self.d_gmm_h0 = None
        self.score = None
        self.likelihood_ratio = None

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
                                  self.y_vec))

            self.residual = self.y_vec - self.design.dot(self.beta)

            for j in np.unique(self.treatment):
                self.inv_variance[self.treatment == j] = 1 / self.residual[
                    self.treatment == j].var()

            i += 1

        design_weight = self.design * self.inv_variance.reshape((-1, 1))
        self.g_score = (design_weight).T.dot(self.y_vec - self.design.dot(
            self.beta)) / self.sample_size
        self.d_g_score = - design_weight.T.dot(self.design) / self.sample_size

        self.v_mat = design_weight.T.dot(self.design) / self.sample_size

        # variance of GMM estimator is
        # \sqrt n (\hat \beta - \beta_0) -> N(0, (G W G^T)^{-1})
        # G = \partial \bar g(beta)^T / \partial \beta

        self.w_mat = np.linalg.inv(self.v_mat)
        self.variance = np.linalg.inv(self.d_g_score.dot(self.w_mat).dot(
            self.d_g_score.T)).diagonal() / self.sample_size

        self.gmm = self.g_score.dot(self.w_mat).dot(self.g_score) * self.sample_size

        # under the H_0
        i = 0
        while i < self.maxiter:
            beta_h0 = np.linalg.solve(self.design_h0.T.dot(
                np.multiply(self.inv_variance.reshape((-1, 1)), self.design_h0)
            ),
                self.design_h0.T.dot(self.inv_variance *
                                     self.y_vec))

            i += 1

        self.beta_h0 = np.zeros(self.design.shape[1])
        self.beta_h0[self.parameter_h0] = beta_h0

        self.g_score_h0 = design_weight.T.dot(
            self.y_vec - self.design.dot(self.beta_h0)) / \
                          self.sample_size

        self.residual_h0 = (self.y_vec - self.design.dot(
            self.beta_h0)).reshape((-1, 1))
        omega_h0 = (design_weight * self.residual_h0).T.dot(design_weight *
                                                            self.residual_h0) / self.sample_size

        d_g_h0 = -design_weight.T.dot(self.design) / self.sample_size

        sigma_h0 = d_g_h0.dot(np.linalg.inv(omega_h0)).dot(
            d_g_h0.T)

        # self.gmm_h0 = self.n * self.g_h0.dot(self.w).dot(self.g_h0)

        gmm_h0 = self.sample_size * self.g_score_h0.dot(np.linalg.inv(omega_h0)).dot(
            self.g_score_h0)

        d_gmm_h0 = d_g_h0.dot(np.linalg.inv(omega_h0)).dot(self.g_score_h0)

        self.score = d_gmm_h0.T.dot(np.linalg.inv(
            sigma_h0)).dot(d_gmm_h0) * self.sample_size

        self.likelihood_ratio = gmm_h0 - self.gmm

    def cuped(self, adjusted_covariate: List):
        """
        variance reduction by CUPED method
        :param adjusted_covariate:
        :return:
        """
        adjusted_covariate = [0] + adjusted_covariate
        theta = np.linalg.solve(
            self.design_h0[:, adjusted_covariate].T.dot(
                self.design_h0[:, adjusted_covariate]),
            self.design_h0[:, adjusted_covariate].T.dot(self.y_vec))

        self.y_cv = self.y_vec - self.design_h0[:, adjusted_covariate].dot(
            theta)

    def l_none_screen(self):
        """
        select relevant covariates using L_0 penalty
        :return:
        """
        pass
