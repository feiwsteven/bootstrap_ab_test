"""

"""
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from rpy2.robjects.packages import importr
from src import stat_func as sf
from src import test_func as tf
from src import bootstraptest as bt
from rpy2.robjects import numpy2ri

rfast = importr('Rfast')
numpy2ri.activate()

if __name__ == '__main__':

    fp = np.zeros(3)
    repeat = 2000
    stat = np.zeros((repeat, 3))
    stat_low = np.zeros((repeat, 3))
    stat_up = np.zeros((repeat, 3))
    p_val = np.zeros((repeat, 3))
    b = 3000
    seed = 11
    np.random.seed(seed)
    for i in range(repeat):
        model = bt.BootstrapTest(np.random.lognormal(2.5, sigma=1, size=1000),
                              np.random.lognormal(2.5, sigma=1, size=1000))

        try:
            stat[i,0], [stat_low[i, 0], stat_up[i,0]], p_val[i, 0] = \
                model.bc_percentile_bootstrap(b, 0.05, 10, sf.mean, tf.ratio)

            stat[i, 1], [stat_low[i, 1],
                         stat_up[i, 1]] = model.t_stat(0.05, sf.mean)

            out = rfast.boot_ttest2(model.x, model.y, b)
            p_val[i, 1] = out[1]

            if stat[i, 0] < stat_low[i, 0] or stat[i, 0] > stat_up[i, 0]:
                fp[0] += 1

            if stat[i, 1] < stat_low[i, 1] or stat[i, 1] > stat_up[i, 1]:
                fp[1] += 1

        except:
            i = i - 1

        if i % 100 == 0:
            logger.info(f"i={i}")

    np.random.seed(seed)
    for i in range(repeat):
        model = bt.BootstrapTest(np.random.lognormal(2.5, sigma=1, size=1000),
                              np.random.lognormal(2.5, sigma=1, size=1000))

        try:
            stat[i, 2], [stat_low[i, 2], stat_up[i, 2]], p_val[i, 2] = \
                model.percentile_t_bootstrap(b, 0.05, 10, sf.mean)
            if stat[i, 2] < stat_low[i, 2] or stat[i, 2] > stat_up[i, 2]:
                fp[2] += 1
        except:
            i = i - 1

        if i % 100 == 0:
            logger.info(f"i={i}")

    fig, axs = plt.subplots(4)
    axs[0].hist(np.random.lognormal(2.5, sigma=1, size=1000),
                bins=100,
                alpha=0.3)
    axs[0].hist(np.random.lognormal(2.5, sigma=1, size=1000),
                bins=100,
                alpha=0.3)

    axs[1].hist(stat[:, 0], bins=100, alpha=0.3, color='blue')
    axs[2].hist(stat[:, 1], bins=100, alpha=0.3, color='yellow')
    axs[3].hist(stat[:, 2], bins=100, alpha=0.3, color='brown')
    plt.tight_layout()
    plt.show()

    print(f"percentile bootstrap = {fp[0]/repeat}, t test = {fp[1]/repeat}, "
          f"percentile-t = {fp[2]/repeat}, R boot-t ={sum(p_val<0.05)/repeat}")
