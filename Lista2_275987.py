import matplotlib.pyplot as plt
import numpy as np
import math as mt
from scipy.stats import uniform, norm, probplot, kstest, geom
import seaborn as sns
from scipy.stats.distributions import cauchy, expon
import scipy.special as sc


"""--------------------- Rozkład normalny -------------------------"""
def normal(n, mu=0, sigma=1):
    """
    Generuje n liczb z rozkładu normalnego o parametrach mu i sigma przy pomocy metody odwrotnej dystrybuanty.

    :param n: (int) liczba liczb do wygenerowania
    :param mu: (float) wartość oczekiwana
    :param sigma: (float) odchylenie standardowe
    :return: (ndarray) n liczb z rozkładu normalnego
    """
    us = uniform.rvs(size=n)
    return norm.ppf(us, loc=mu, scale=sigma)


def normal_pdf_plot(mu=0, sigma=1):
    X = normal(1000, mu, sigma)
    xs = np.arange(mu - 3*sigma, mu+3*sigma, 0.001)
    plt.plot(xs, [norm.pdf(x, loc=mu, scale=sigma) for x in xs], 'y-', label='pdf')
    sns.histplot(X, stat='density', label='empirical')
    plt.xlim(mu - 3*sigma, mu+3*sigma)
    plt.legend()
    plt.show()


def normal_cdf_plot(mu=0, sigma=1):
    X = normal(1000, mu, sigma)
    xs = np.arange(mu - 3*sigma, mu+3*sigma, 0.001)
    plt.plot(xs, [norm.cdf(x, loc=mu, scale=sigma) for x in xs], 'y-', label='pdf')
    sns.ecdfplot(X, label='empirical')
    plt.legend()
    plt.show()


def normal_qqplot(mu=0, sigma=1):
    X = normal(10000, mu, sigma)
    probplot(X, plot=plt, dist='norm', sparams=(mu, sigma))
    plt.show()


def normal_stats(mu=0, sigma=1):
    X = normal(1000, mu=mu, sigma=sigma)
    print('Expected value empirical:', np.mean(X), 'Expected value theoritical:', mu)
    print('Variance empirical:', np.var(X), 'Variance theoritical:', sigma**2)
    alpha = 0.05
    p_val = kstest(X, norm.cdf, args=(mu, sigma)).pvalue
    print("P-value:", p_val)
    print("Null hypothesis rejected:", p_val < alpha)


"""--------------------- Rozkład wykładniczy -------------------------"""""
def exp(lamb, n=1):
    us = uniform.rvs(size=n)
    xs = [-1 / lamb * np.log(u) for u in us]
    return xs


def exp_inv(lamb, n=1):
    us = uniform.rvs(size=n)
    xs = F_inv(lambda x: 1 - np.exp(-lamb * x), us)
    # F_inv(lambda x: x ** 2, xs)
    return xs


def exp_pdf_comparison(lamb=1):
    X1 = exp(lamb, 1000)
    X2 = exp_inv(lamb, 1000)
    sns.histplot(X1, stat='density', label='empirical manually')
    sns.histplot(X2, stat='density', label='empirical N-R')
    plt.xlim(0, 10)
    plt.legend()
    plt.show()


def exp_pdf_plot(lamb=1):
    X = exp(lamb, 1000)
    xs = np.arange(0, 10, 0.001)
    plt.plot(xs, [lamb * np.exp(-lamb * x) for x in xs], 'y-', label='pdf')
    sns.histplot(X, stat='density', label='empirical')
    plt.xlim(0, 10)
    plt.legend()
    plt.show()


def exp_cdf_plot(lamb=1):
    X = exp(lamb, 1000)
    xs = np.arange(0, 10, 0.001)
    sns.ecdfplot(X, label='empirical')
    plt.plot(xs, [1 - np.exp(-lamb * x) for x in xs], 'y-', label='pdf', alpha=0.7)
    plt.legend()
    plt.show()


def exp_qqplot(lamb=1):
    X = exp(lamb, 1000)
    probplot(X, plot=plt, dist='expon', sparams=(0, 1/lamb))
    plt.show()


def exp_stats(lamb=1):
    X = exp(lamb, 1000)
    print('Expected value empirical:', np.mean(X), 'Expected value theoritical:', 1/lamb)
    print('Variance empirical:', np.var(X), 'Variance theoritical:', 1/lamb**2)
    alpha = 0.05
    p_val = kstest(X, expon.cdf, args=(0, 1/lamb)).pvalue
    print("P-value:", p_val)
    print("Null hypothesis rejected:", p_val < alpha)


"""---------------------------- Cauchy ----------------------------"""
def cauchys(n, x0=0, gamma=1):
    us = uniform.rvs(size=n)
    return gamma * np.tan(np.pi * (us - 0.5)) + x0


def cauchy_pdf_plot(x0=0, gamma=1):
    X = cauchys(1000, x0, gamma)
    xs = np.arange(-10, 10, 0.001)
    plt.plot(xs, [1 / (np.pi * gamma * (1 + ((x-x0) / gamma)**2)) for x in xs], 'y-', label='pdf')
    sns.histplot(X, stat='density', label='empirical')
    plt.xlim(-10, 10)
    plt.legend()
    plt.show()


def cauchy_cdf_plot(x0=0, gamma=1):
    X = cauchys(1000, x0, gamma)
    xs = np.arange(-10, 10, 0.001)
    sns.ecdfplot(X, label='empirical')
    plt.plot(xs, [1 / np.pi * np.arctan((x-x0) / gamma) + 0.5 for x in xs], label='cdf', c='red', linestyle='--')
    plt.xlim(-4, 4)
    plt.legend()
    plt.show()


def cauchy_qqplot(x0=0, gamma=1):
    X = cauchys(1000, x0=x0, gamma=gamma)
    probplot(X, plot=plt, dist='cauchy', sparams=(x0, gamma))
    plt.show()


def cauchy_stats(x0=0, gamma=1):
    X = cauchys(100000, x0=x0, gamma=gamma)
    print('Expected value empirical:', np.mean(X), 'Expected value theoritical:', x0)
    print('Variance empirical:', np.var(X), 'Variance theoritical:', np.inf)
    alpha = 0.05
    p_val = kstest(X, 'cauchy', args=(x0, gamma)).pvalue
    print("P-value:", p_val)
    print("Null hypothesis rejected:", p_val < alpha)


"""--------------------- Rozkład geometryczny -------------------------"""
def geo(n, p=0.5):
    us = uniform.rvs(size=n)
    return np.ceil(np.log(us) / np.log(1-p))


def geom_pmf_plot(p=0.5):
    X = geo(1000, p)
    xs = np.arange(1, np.floor(max(X)), 1)
    sns.histplot(X, stat='probability', label='empirical', binwidth=1)
    plt.scatter(x=xs, y=[(1-p)**(x-1) * p for x in xs], c='y', label='pmf', alpha=0.5)
    plt.legend()
    plt.show()


def geom_cdf_plot(p=0.5):
    X = geo(1000, p)
    xs = np.arange(1, np.floor(max(X)), 0.001)
    sns.ecdfplot(X, label='empirical')
    plt.plot(xs, [1 - (1-p)**np.floor(x) for x in xs], c='red', linestyle='--', label='cdf', alpha=0.6)
    plt.legend()
    plt.show()


def geom_qqplot(p=0.5):
    X = geo(1000, p)
    probplot(X, plot=plt, dist='geom', sparams=(p,))
    plt.show()


def geom_stats(p=0.5):
    X = geo(1000, p)
    print('Expected value empirical:', np.mean(X), 'Expected value theoritical:', 1/p)
    print('Variance empirical:', np.var(X), 'Variance theoritical:', (1-p) / p**2)
    geo_dist = geom(p)
    alpha = 0.05
    p_val = kstest(X, 'geom', args=(p,)).pvalue
    print("P-value:", p_val)
    print("Null hypothesis rejected:", p_val < alpha)


"""--------------------- Rozkład Poissona -------------------------"""
def poiss(n, lamb=1):
    us = uniform.rvs(size=n)
    xs = np.zeros(n)
    for j in range(n):
        u = us[j]
        i = 0
        p = np.exp(-lamb)
        F = p
        while u > F:
            p *= lamb / (i+1)
            F += p
            i += 1
        xs[j] = i
    return xs


def poiss_pmf_plot(lamb=1):
    X = poiss(1000, lamb)
    xs = np.arange(0, np.floor(max(X)), 1)
    sns.histplot(X, stat='probability', label='empirical', binwidth=1, alpha=0.5)
    plt.scatter(xs, [np.exp(-lamb) * lamb ** x / mt.gamma(x + 1) for x in xs], c='y', label='pmf')
    plt.legend()
    plt.show()


def poiss_cdf_plot(lamb=1):
    X = poiss(1000, lamb)
    xs = np.arange(0, np.floor(max(X)), 0.001)
    plt.plot(xs, [sc.gammaincc(np.floor(x+1), lamb) for x in xs], 'y-', label='cdf')
    sns.ecdfplot(X, label='empirical')
    plt.legend()
    plt.show()


def poiss_qqplot(lamb=1):
    X = poiss(1000, lamb)
    probplot(X, plot=plt, dist='poisson', sparams=(lamb,))
    plt.show()


def poiss_stats(lamb=1):
    X = poiss(1000, lamb)
    print('Expected value empirical:', np.mean(X), 'Expected value theoritical:', lamb)
    print('Variance empirical:', np.var(X), 'Variance theoritical:', lamb)
    alpha = 0.05
    p_val = kstest(X, 'poisson', args=(lamb,)).pvalue
    print("P-value:", p_val)
    print("Null hypothesis rejected:", p_val < alpha)


"""---------------------------------Table lookup____________________________________________________"""
def table_lookup(values, probabilities, n=1):
    us = uniform.rvs(size=n)
    bins = np.cumsum(probabilities)
    indices = np.digitize(us, bins)
    return np.array(values)[indices]


def tl_pmf_plot(values, probabilities, n=1000):
    X = table_lookup(values, probabilities, n)
    val, cnt = np.unique(X, return_counts=True)
    pmf = cnt / len(X)
    plt.bar(val, pmf, alpha=0.5, label='empirical')
    plt.scatter(values, probabilities, c='y', label='pmf')
    plt.legend()
    plt.show()


def tl_cdf_plot(values, probabilities):
    X = table_lookup(values, probabilities, 1000)
    plt.scatter(values, np.cumsum(probabilities), c='y', label='cdf')
    sns.ecdfplot(X, label='empirical')
    plt.legend()
    plt.show()


def tl_stats(values, probabilities):
    values, probabilities = np.array(values), np.array(probabilities)
    X = table_lookup(values, probabilities, 1000)
    ex = np.dot(values, probabilities)
    var = np.dot(values ** 2, probabilities) - ex ** 2
    print('Expected value empirical:', np.mean(X), 'Expected value theoritical:', ex)
    print('Variance empirical:', np.var(X), 'Variance theoritical:', var)


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




if __name__ == '__main__':
    """--------------------- Rozkład normalny -------------------------"""
    # normal_pdf_plot(3)
    # normal_cdf_plot(2)
    # normal_qqplot()
    # normal_stats()

    """--------------------- Rozkład wykładniczy -------------------------"""
    # exp_pdf_plot(2)
    # exp_cdf_plot()
    # exp_qqplot()
    # exp_stats(2)

    """--------------------- Rozkład cauchyego -------------------------"""
    # cauchy_pdf_plot(2, 0.5)
    # cauchy_cdf_plot(-2)
    # cauchy_qqplot()
    # cauchy_stats(-2)

    """--------------------- Rozkład geometryczny -------------------------"""
    # geom_pmf_plot(0.1)
    # geom_cdf_plot(0.2)
    # geom_qqplot(0.1)
    # geom_stats(0.5)  # null hypothesis rejected?

    """--------------------- Rozkład poissona -------------------------"""
    # poiss_pmf_plot(2)
    # poiss_cdf_plot(3)
    # poiss_qqplot(3)
    # poiss_stats(3)   # null hypothesis rejected?

    """--------------------- Table lookup -------------------------"""
    vals, probs = [1, 2, 3, 4], 4 * [0.25]

    # tl_pmf_plot(vals, probs, 10000)
    # tl_cdf_plot(vals, probs)
    # tl_stats(vals, probs)

    """-----------------------F_inv--------------------------------"""
    # xs = np.arange(0, 10, 0.1)
    # plt.plot(xs, F_inv(lambda x: x**2, xs))
    # plt.show()
    exp_pdf_comparison(3)

    pass
