"""
A simulation study to compare T-test and GMM test under several different
specifications.
"""
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from src.covariate_adjusted import CovariateAdjustedAncova
from scipy.stats import probplot

from scipy.stats import norm, chi2, ttest_ind

if __name__ == '__main__':
    N_REP = 5000

    beta_cv = np.zeros((N_REP, 1))
    lr = np.zeros(N_REP)
    score = np.zeros(N_REP)
    variance = np.zeros((N_REP, 5))
    N = 5000
    np.random.seed(10)
    pv = np.zeros(N_REP)
    pv_cv = np.zeros(N_REP)

    N_TREATMENT = 3
    pi = [1/2, 1/4, 1/4]
    effect = [0, 0.3, 0.0]
    var = [0.2, 0.7, 0.8]
    beta = np.zeros((N_REP, 3 + N_TREATMENT))

    variance = np.zeros((N_REP, 3 + N_TREATMENT))

    for i in range(N_REP):
        x = np.random.normal(10, 1, 4 * N).reshape((N, -1))

        treatment = np.random.multinomial(N_TREATMENT - 1, pi, size=N)[:, 0]

        y = np.zeros(N)
        for j in range(3):
            epsilon = np.random.normal(0, var[j], np.sum(treatment == j))
            epsilon = epsilon - epsilon.mean()

            y[treatment == j] = 1 + effect[j] + (x[treatment == j,]).dot(
            [1, 2, 3,
                                                                        4]) + epsilon

        if i % 500 == 0:
            logger.info(i)

        # change parameter_h0 according to H_0
        parameter_h0 = [0, 1, 2, 3]
        model = CovariateAdjustedAncova(x[:, :3], y, treatment, 5, parameter_h0)
        model.fit()
        model.cuped([1])
        y_cv = model.y_cv
        beta[i,] = model.beta
        lr[i] = model.likelihood_ratio
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
    probplot(score, dist=chi2, sparams=2, plot=plt)
    plt.show()
    logger.info(f"pv={np.sum(pv < 0.05) / N_REP}, pv_cv="
                f"{np.sum(pv_cv < 0.05) / N_REP}, score="
                f"{np.sum(score > chi2.ppf(0.95, df=2)) / N_REP}, lr"
                f"={np.sum(lr > chi2.ppf(0.95, df=2)) / N_REP}")
