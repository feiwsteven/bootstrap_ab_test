import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from src.bootstraptest_ancova import BootstrapAncova

from scipy.stats import norm, chi2, ttest_ind

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