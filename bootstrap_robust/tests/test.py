import unittest

import numpy as np
from loguru import logger
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr

import bootstrap_robust.src.bootstraptest as bt
import bootstrap_robust.src.stat_func as sf

rfast = importr('Rfast')
numpy2ri.activate()


class TestBootstrapTest(unittest.TestCase):
    def test_lognormal(self):
        fp = np.zeros(3)
        repeat = 500
        stat = np.zeros((repeat, 3))
        stat_low = np.zeros((repeat, 3))
        stat_up = np.zeros((repeat, 3))
        p_val = np.zeros((repeat, 3))
        b = 2000
        seed = 11
        np.random.seed(seed)
        for i in range(repeat):
            model = bt.BootstrapTest(
                np.random.lognormal(2.5, sigma=1, size=100),
                np.random.lognormal(2.5, sigma=1, size=100))

            try:
                stat[i, 0], [stat_low[i, 0], stat_up[i, 0]], p_val[i, 0] = \
                    model.bc_percentile_bootstrap(b, 0.05, 10, sf.mean)

                stat[i, 1], [stat_low[i, 1],
                             stat_up[i, 1]] = model.t_stat(0.05, sf.mean)

                out = rfast.boot_ttest2(model.x, model.y, b)
                p_val[i, 1] = out[1]

                if stat[i, 0] < stat_low[i, 0] or stat[i, 0] > stat_up[i, 0]:
                    fp[0] += 1

                if stat[i, 1] < stat_low[i, 1] or stat[i, 1] > stat_up[i, 1]:
                    fp[1] += 1

                if stat[i, 2] < stat_low[i, 2] or stat[i, 2] > stat_up[i, 2]:
                    fp[2] += 1
            except:
                i = i - 1

            if i % 100 == 0:
                logger.info(f"i={i}")

        np.random.seed(seed)
        for i in range(repeat):
            model = bt.BootstrapTest(
                np.random.lognormal(2.5, sigma=1, size=100),
                np.random.lognormal(2.5, sigma=1, size=100))

            try:
                stat[i, 2], [stat_low[i, 2], stat_up[i, 2]], p_val[i, 2] = \
                    model.percentile_t_bootstrap(b, 0.05, 10, sf.mean)
                if stat[i, 2] < stat_low[i, 2] or stat[i, 2] > stat_up[i, 2]:
                    fp[2] += 1
            except:
                i = i - 1

            if i % 100 == 0:
                logger.info(f"i={i}")

        print(
            f"percentile bootstrap = {fp[0] / repeat}, t test = {fp[1] / repeat}, "
            f"percentile-t = {fp[2] / repeat}, R boot-t ={sum(p_val < 0.05) / repeat}"
        )

        self.assertGreater(1 - np.sum(p_val[:,0] < 0.05) / repeat, 1 - 0.06,
                           'type I error is inflated.')


    def test_normal(self):
        fp = np.zeros(3)
        repeat = 500
        stat = np.zeros((repeat, 3))
        stat_low = np.zeros((repeat, 3))
        stat_up = np.zeros((repeat, 3))
        p_val = np.zeros((repeat, 3))
        b = 2000
        seed = 11
        np.random.seed(seed)
        for i in range(repeat):
            model = bt.BootstrapTest(
                np.random.normal(2.5, scale=1, size=100),
                np.random.normal(2.5, scale=1, size=100))

            try:
                stat[i, 0], [stat_low[i, 0], stat_up[i, 0]], p_val[i, 0] = \
                    model.bc_percentile_bootstrap(b, 0.05, 10, sf.mean)

                stat[i, 1], [stat_low[i, 1],
                             stat_up[i, 1]] = model.t_stat(0.05, sf.mean)

                out = rfast.boot_ttest2(model.x, model.y, b)
                p_val[i, 1] = out[1]

                if stat[i, 0] < stat_low[i, 0] or stat[i, 0] > stat_up[i, 0]:
                    fp[0] += 1

                if stat[i, 1] < stat_low[i, 1] or stat[i, 1] > stat_up[i, 1]:
                    fp[1] += 1

                if stat[i, 2] < stat_low[i, 2] or stat[i, 2] > stat_up[i, 2]:
                    fp[2] += 1
            except:
                i = i - 1

            if i % 100 == 0:
                logger.info(f"i={i}")

        np.random.seed(seed)
        for i in range(repeat):
            model = bt.BootstrapTest(
                np.random.normal(2.5, scale=1, size=100),
                np.random.normal(2.5, scale=1, size=100))

            try:
                stat[i, 2], [stat_low[i, 2], stat_up[i, 2]], p_val[i, 2] = \
                    model.percentile_t_bootstrap(b, 0.05, 10, sf.mean)
                if stat[i, 2] < stat_low[i, 2] or stat[i, 2] > stat_up[i, 2]:
                    fp[2] += 1
            except:
                i = i - 1

            if i % 100 == 0:
                logger.info(f"i={i}")

        print(
            f"percentile bootstrap = {fp[0] / repeat}, t test = {fp[1] / repeat}, "
            f"percentile-t = {fp[2] / repeat}, R boot-t ={sum(p_val < 0.05) / repeat}"
        )

        self.assertGreater(1 - np.sum(p_val[:,0] < 0.05) / repeat, 1 - 0.06,
                           'type I error is inflated.')

