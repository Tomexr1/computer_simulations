import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import uniform, norm, probplot, kstest, geom
import scipy.stats as stats
import scipy.special as sp


def newton_method(f, x0, dx=0.0001):
    """
    Znajduje miejsce zerowe funkcji f metodą Newtona.

    :param f: funkcja
    :param x0: wartość startowa
    :return: przybliżenie miejsca zerowego
    """
    x = x0
    for _ in range(100):
        x = x - f(x) / (f(x + dx) - f(x)) * dx
    return x


def F_inv(F, us, dx=0.001):
    return [newton_method(lambda y: F(y) - u, 0, dx=dx) for u in us]


def drawQQ(X, distr):
    # X = sorted(X)
    X.sort()
    N = len(X)
    Q = F_inv(distr, np.arange(1, N+1) / N)
    plt.scatter(Q, X)
    plt.plot([min(X), max(X)], [min(X), max(X)], 'r-', alpha=0.5)
    plt.show()





def X_discrete(n=1):
    Xs = []
    Xs_dict = {1: 0.11, 2: 0.12, 3:0.27, 4: 0.19, 5: 0.31}
    c = 1.55
    for i in range(n):
        while True:
            Y = np.ceil(5*uniform.rvs())
            u = uniform.rvs()
            if u <= 5/c * Xs_dict[Y]:
                Xs.append(Y)
                break
    return Xs


def X_disc_cdf_formula(x):
    if x < 1:
        return 0
    elif 1 <= x < 2:
        return 0.11
    elif 2 <= x < 3:
        return 0.11 + 0.12
    elif 3 <= x < 4:
        return 0.11 + 0.12 + 0.27
    elif 4 <= x < 5:
        return 0.11 + 0.12 + 0.27 + 0.19
    elif 5 <= x < 6:
        return 0.11 + 0.12 + 0.27 + 0.19 + 0.31
    else:
        return 1


def X_disc_plots_stats():
    Xs_dict = {1: 0.11, 2: 0.12, 3: 0.27, 4: 0.19, 5: 0.31}
    X = X_discrete(10000)
    print('Expected value: empirical:', np.average(X), 'theoretical:', sum([k * v for k, v in Xs_dict.items()]))
    print('Variance: empirical:', np.var(X), 'theoretical:', sum([k ** 2 * v for k, v in Xs_dict.items()]) - sum([k * v for k, v in Xs_dict.items()]) ** 2)
    val, cnt = np.unique(X, return_counts=True)
    pmf = cnt / len(X)
    fig, ax = plt.subplots(2, 1, layout='constrained')
    ax[0].vlines(val,0, pmf, alpha=0.5, label='pmf empirical')
    ax[0].scatter(Xs_dict.keys(), Xs_dict.values(), c='y', label='pmf theoretical')
    ax[1].plot(np.arange(0, 7, 0.01), [X_disc_cdf_formula(x) for x in np.arange(0, 7, 0.01)],
             label='cdf theoretical', linestyle=(0, (5, 10)))
    sns.ecdfplot(X, label='cdf empirical', color='red', alpha=0.5, ax=ax[1])
    fig.legend()
    plt.show()


def gamma_3_2(n=1):
    Xs = np.zeros(n)
    f = lambda x: 1 / 16 * x ** 2 * np.exp(-x / 2)
    g = lambda x: 1 / 16 * np.exp(-x / 16)  # expon(1/16)
    c = 1024 / 49 / np.e ** 2  # ~ 2.83, policzone pochodną z f/g
    i = 0
    while i < n:
        Y = -16 * np.log(np.random.uniform())
        U = np.random.uniform()
        if U <= f(Y) / (g(Y)) / c:
            Xs[i] = Y
            i += 1
    return Xs


def gamma_plots_stats():
    X = gamma_3_2(10000)
    print('Expected value: empirical:', np.average(X), 'theoretical:', 3*2)
    print('Variance: empirical:', np.var(X), 'theoretical:', 3*2**2)
    fig, ax = plt.subplots(3, 1, layout='constrained')
    val = np.unique(X)
    ax[0].plot(val, [1 / 16 * x ** 2 * np.exp(-x / 2) for x in val], c='y', label='pdf theoretical')
    sns.histplot(X, stat='density', ax=ax[0], color='red', alpha=0.5, label='pdf empirical')
    ax[1].plot(np.arange(0, 20, 0.01), stats.gamma.cdf(np.arange(0, 20, 0.01), a=3, scale=2),
             label='cdf theoretical', linestyle=(0, (5, 10)))
    sns.ecdfplot(X, label='cdf empirical', color='red', alpha=0.5, ax=ax[1])
    probplot(X, dist='gamma', sparams=(3, 2), plot=ax[2])
    fig.legend()
    plt.show()


def X1(n=1):
    Xs = []
    f = lambda x: 3 / 2 * (1 - x**2)
    m = 1.5
    for _ in range(n):
        while True:
            u1, u2 = np.random.uniform(0, 1), np.random.uniform(0, m)
            if u2 <= f(u1):
                Xs.append(u1)
                break
    return Xs


def X1_plots_stats():
    X = X1(1000)
    f = lambda x: 3 / 2 * (1 - x ** 2)
    F = lambda x: -1 / 2 * x * (x**2 - 3)
    print('Expected value: empirical:', np.average(X), 'theoretical:', 0.375)
    print('Variance: empirical:', np.var(X), 'theoretical:', 19 / 320)
    fig, ax = plt.subplots(3, 1, layout='constrained')
    val = np.unique(X)
    ax[0].plot(val, [f(x) for x in val], c='y', label='pdf theoretical')
    sns.histplot(X, stat='density', ax=ax[0], color='red', alpha=0.5, label='pdf empirical')
    ax[1].plot(val, [F(x) for x in val],
               label='cdf theoretical', linestyle=(0, (5, 10)))
    sns.ecdfplot(X, label='cdf empirical', color='red', alpha=0.5, ax=ax[1])
    X.sort()
    N = len(X)
    Q = F_inv(F, np.arange(1, N + 1) / N)
    ax[2].scatter(Q, X, s=1)
    ax[2].plot([min(X), max(X)], [min(X), max(X)], 'r-', alpha=0.5)
    ax[2].set_ylabel('QQ plot')
    fig.legend()
    plt.show()


def X2(n=1):
    Xs = []
    f = lambda x: 3 / 2 * np.sin(x) * (np.cos(x))**2  # on [0, pi] = 3/4 * sin(2x)
    m = 1 / np.sqrt(3)
    for _ in range(n):
        while True:
            u1, u2 = np.random.uniform(0, np.pi), np.random.uniform(0, m)
            if u2 <= f(u1):
                Xs.append(u1)
                break
    return Xs


def X2_plots_stats():
    X = X2(1000)
    f = lambda x: 3 / 2 * np.sin(x) * (np.cos(x))**2
    F = lambda x: 1 / 2 * (1 - (np.cos(x))**3)
    print('Expected value: empirical:', np.average(X), 'theoretical:', np.pi / 2)
    print('Variance: empirical:', np.var(X), 'theoretical:', np.pi**2 / 4 - 14/9)
    fig, ax = plt.subplots(3, 1, layout='constrained')
    val = np.unique(X)
    ax[0].plot(val, [f(x) for x in val], c='y', label='pdf theoretical')
    sns.histplot(X, stat='density', ax=ax[0], color='red', alpha=0.5, label='pdf empirical')
    ax[1].plot(val, [F(x) for x in val],
               label='cdf theoretical', linestyle=(0, (5, 10)))
    sns.ecdfplot(X, label='cdf empirical', color='red', alpha=0.5, ax=ax[1])
    X.sort()
    N = len(X)
    Q = F_inv(F, np.arange(1, N + 1) / N)
    ax[2].scatter(Q, X, s=1)
    ax[2].plot([min(X), max(X)], [min(X), max(X)], 'r-', alpha=0.5)
    ax[2].set_ylabel('QQ plot')
    fig.legend()
    plt.show()


def X3(n=1):
    Xs = []
    f = lambda x: 2 * np.sqrt(1 / 2 / np.pi) * np.exp(-x ** 2 / 2)
    g = lambda x: np.exp(-x) # expon(1)
    c = np.sqrt(2 * np.e / np.pi)  # policzone pochodną z f/g
    for _ in range(n):
        while True:
            Y = -1 * np.log(np.random.uniform())
            U = np.random.uniform()
            if U <= f(Y) / (g(Y)) / c:
                Xs.append(Y)
                break
    return Xs


def X3_plots_stats():
    X = X3(1000)
    print('Expected value: empirical:', np.average(X), 'theoretical:', np.sqrt(2/np.pi))
    print('Variance: empirical:', np.var(X), 'theoretical:', 1 - 2/np.pi)
    fig, ax = plt.subplots(3, 1, layout='constrained')
    val = np.unique(X)
    f = lambda x: 2 * np.sqrt(1 / 2 / np.pi) * np.exp(-x ** 2 / 2)
    F = lambda x: sp.erf(x / np.sqrt(2))
    ax[0].plot(val, [f(x) for x in val], c='y', label='pdf theoretical')
    sns.histplot(X, stat='density', ax=ax[0], color='red', alpha=0.5, label='pdf empirical')
    ax[1].plot(val, [F(x) for x in val],
             label='cdf theoretical', linestyle=(0, (5, 10)))
    sns.ecdfplot(X, label='cdf empirical', color='red', alpha=0.5, ax=ax[1])
    X.sort()
    N = len(X)
    Q = F_inv(F, np.arange(1, N + 1) / N)
    ax[2].scatter(Q, X, s=1)
    ax[2].plot([min(X), max(X)], [min(X), max(X)], 'r-', alpha=0.5)
    ax[2].set_ylabel('QQ plot')
    fig.legend()
    plt.show()


if __name__ == '__main__':
    # X_disc_plots_stats()
    gamma_plots_stats()
    # X1_plots_stats()
    # X2_plots_stats()
    # X3_plots_stats()






    pass
