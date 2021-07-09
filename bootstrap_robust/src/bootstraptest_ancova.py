import numpy as np
from scipy.stats import norm
import bootstrap_robust.src.stat_func as sf
import bootstrap_robust.src.test_func as tf

__version__ = '0.1'


class BootstrapAncova:
    def __init__(self, x: np.ndarray, y: np.ndarray, treatment: np.ndarray):
        self.x = x
        self.y = y
        self.treatment = treatment
        self.n = y.shape[0]

        self.x_centered = self.x - self.x.mean(axis=0)
        self.y_centered = self.y - self.y.mean(axis=0)
