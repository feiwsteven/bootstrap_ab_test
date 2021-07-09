import numpy as np
from scipy.stats import norm
import bootstrap_robust.src.stat_func as sf
import bootstrap_robust.src.test_func as tf

__version__ = '0.1'


class BootstrapTest:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

        self.n_x = x.shape[0]
        self.n_y = y.shape[0]
        self.normalized_x = self.x - self.x.mean() + (self.x.sum() + \
                                                      self.y.sum()) / (
                                        self.n_x + self.n_y)
        self.normalized_y = self.y - self.y.mean() + (self.x.sum() + \
                                                      self.y.sum()) / (
                                        self.n_x + self.n_y)

    def define_test_metric(self):
        pass

    def t_stat(self, type_1, stat_func=sf.mean):
        t = (stat_func(self.x) - stat_func(self.y)) / (self.x.var(ddof=1) /
                                                        self.n_x +
                                               self.y.var(ddof=1) / self.n_y)\
            ** 0.5
        return t, norm.ppf([type_1 / 2, 1 - type_1 / 2])

    def iter_i(self, i):
        # This is centered percentile src
        x_b = np.random.choice(self.normalized_x, int(self.n_x ** 0.5),
                               replace=True)
        y_b = np.random.choice(self.normalized_y, int(self.n_y ** 0.5),
                               replace=True)
        return x_b, y_b

    def bc_percentile_bootstrap(self, b: int, type_1: float, seed: int,
                                stat_func=sf.mean, test_func=tf.difference):
        """
        Bias-corrected percentile src
        """
        stat = test_func(stat_func(self.x), stat_func(self.y))

        stat_star = np.zeros(b)

        for i in range(b):
            x_b, y_b = self.iter_i(i)
            stat_star[i] = test_func(stat_func(x_b), stat_func(y_b)) * \
                           np.sqrt(x_b.shape[0] / self.n_x)

        p_0 = (stat_star <= stat).sum() / b
        z_0 = norm.ppf(p_0)
        low_p = norm.cdf(2 * z_0 + norm.ppf(type_1 / 2))
        up_p = norm.cdf(2 * z_0 + norm.ppf(1 - type_1 / 2))
        p_val = np.sum(np.abs(stat_star) >= np.abs(stat))/ b


        return stat, np.sort(stat_star)[[int(low_p * b), int(up_p * b)]], p_val

    def percentile_t_bootstrap(self, b: int, type_1: float, seed: int,
                               stat_func=sf.mean):
        """
        Another name is src-t method.
        """

        stat = (stat_func(self.x) - stat_func(self.y)) / (self.x.var(ddof=1) /
                                                        self.n_x +
        self.y.var(ddof=1) / self.n_y) ** 0.5

        stat_star = np.zeros(b)

        for i in range(b):
            x_b, y_b = self.iter_i(i)

            stat_star[i] = (stat_func(x_b) - stat_func(y_b)) / np.sqrt(
                    x_b.var(ddof=1) / self.n_x +
                                                      y_b.var(ddof=1) /
                                       self.n_y) * np.sqrt(x_b.shape[0]
                                                                 / self.n_x)

        low_p = type_1 / 2
        up_p = 1 - type_1 / 2
        p_val = np.sum(np.abs(stat_star) >= np.abs(stat)) / b
        # TODO the lower and upper percentiles need to be further modified.
        return stat, np.sort(stat_star)[[int(low_p * b), int(up_p * b)]], p_val
