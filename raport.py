import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.special import erf
import scipy.stats as stats
import seaborn as sns

def ACORN(N, k=9, M=2**89-1, Lag=1000):
    # N - number of samples
    # k - number of bits
    # M - modulus
    # Lag - lag
    # returns: list of N samples
    seed = np.random.randint(0, 2**32)
    Xs = np.zeros((k+1, N+Lag))
    Xs[0, :] = seed
    # for i in range(1, N+Lag):
        # Xs[0, i] = (Xs[0, i-1] + Xs[1, i-1]) % M
        # for j in range(1, k):
        #     Xs[j, i] = Xs[j-1, i-1]
    for m in range(1, k+1):
        for n in range(1, N+Lag):
            Xs[m, n] = (Xs[m-1, n] + Xs[m, n-1]) % M
    return Xs[k, Lag:] / M



def ACORN_plot(N, k=9, M=2**89-1, Lag=1000):
    Xs = ACORN(N, k, M, Lag)
    fig, ax = plt.subplots()
    ax.plot(Xs[:N//2], Xs[N//2:], 'o')
    plt.show()



# ACORN_plot(10000)
# print(ACORN(1000))
# Xs = ACORN(1000)
# # Xs from 2nd value to second to last
# print(len(Xs[1:]))

def acorn_vs_numpy(N, k=9, M=2**89-1, Lag=1000):
    import time
    Ns = np.arange(100, N, 100)
    acorn_times, numpy_times = [], []
    for n in Ns:
        start = time.time()
        ACORN(n, k, M, Lag)
        acorn_times.append(time.time() - start)
        start = time.time()
        np.random.rand(n)
        numpy_times.append(time.time() - start)
    fig, ax = plt.subplots()
    ax.scatter(Ns, acorn_times, label='ACORN', s=1)
    ax.scatter(Ns, numpy_times, label='numpy', s=1)
    ax.set_xlabel('n')
    ax.set_ylabel('time')
    ax.legend()
    plt.show()



# acorn_vs_numpy(10000)

def tuzin(n, mu=0, sigma=1):
    Us = np.random.uniform(size=(n, 12))
    return sigma * (np.sum(Us, axis=1) - 6) + mu


def ziggurat(size, mu=0, sigma=1):

    f = lambda x: np.exp(-x**2 / 2)
    f_inv = lambda y: np.sqrt(-2 * np.log(y))
    n = 256
    x1 = 3.6541528853610088
    y1 = f(x1)
    Table = np.zeros(shape=(n, 2))
    Table[0, 0] = x1
    Table[0, 1] = y1
    tail_area = integrate.quad(f, x1, np.inf)[0]
    A = x1 * y1 + tail_area
    for i in range(1, n-1):
        Table[i, 1] = Table[i-1, 1] + A / Table[i-1, 0]
        Table[i, 0] = f_inv(Table[i, 1])
    Table[n - 1, 0] = 0
    Table[n - 1, 1] = 1
    print(Table.shape)

    Xs = np.zeros(size)
    for i in range(size):
        while True:
            indx = np.floor(256 * np.random.uniform() - 1)
            U0 = 2 * np.random.uniform() - 1
            if i == -1:
                x = U0 * A / Table[0, 1]
            else:
                x = U0 * Table[indx, 0]
            if np.abs(x) < Table[indx+1, 0]:
                Xs[i] = x
                break
            if indx == -1:  # fallback
                while True:
                    x = -np.log(np.random.uniform()) / Table[0, 0]
                    y = -np.log(np.random.uniform())
                    if 2 * y > x**2:
                        Xs[i] = x + Table[0, 0]
                        break
                break
            else:
                y = Table[indx, 1] + np.random.uniform() * (Table[indx+1, 1] - Table[indx, 1])
                if y < f(np.abs(x)):
                    Xs[i] = x
                    break
    return Xs * sigma + mu


def norm_inv_cdf(n, mu=0, sigma=1):
    """
    Generuje n liczb z rozkładu normalnego o parametrach mu i sigma przy pomocy metody odwrotnej dystrybuanty.

    :param n: (int) liczba liczb do wygenerowania
    :param mu: (float) wartość oczekiwana
    :param sigma: (float) odchylenie standardowe
    :return: (np.ndarray) n liczb z rozkładu normalnego
    """

    us = stats.uniform.rvs(size=n)
    return stats.norm.ppf(us, loc=mu, scale=sigma)


def box_muller(n=1, mu=0, sigma=1):
    """
    Generuje n liczb z rozkładu normalnego o parametrach mu i sigma za pomocą metody Boxa-Muellera

    :param n: (int) liczba liczb do wygenerowania
    :param mu: (float) wartość oczekiwana
    :param sigma: (float) odchylenie standardowe
    :return: (np.ndarray) n liczb z rozkładu normalnego
    """

    u1, u2 = np.random.uniform(size=int(np.ceil(n / 2))), np.random.uniform(
        size=int(np.ceil(n / 2))
    )
    Xs = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2) * sigma + mu
    Ys = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2) * sigma + mu
    return np.hstack((Xs, Ys))[:n]


def polar_method(n=1, mu=0, sigma=1):
    """
    Generuje n liczb z rozkładu normalnego o parametrach mu i sigma za pomocą metody biegunowej

    :param n: (int) liczba liczb do wygenerowania
    :param mu: (float) wartość oczekiwana
    :param sigma: (float) odchylenie standardowe
    :return: (np.ndarray) n liczb z rozkładu normalnego
    """

    Zs = []
    while True:
        us = np.array(
            [
                np.random.uniform(-1, 1, size=int(np.ceil(0.6 * (n - len(Zs))))),
                np.random.uniform(-1, 1, size=int(np.ceil(0.6 * (n - len(Zs))))),
            ]
        )  # expected value powtórzeń ~ 1.2
        R_sq = np.sum(us ** 2, axis=0)
        us = np.vstack((us, R_sq))
        xs = np.where(
            us[2] <= 1, np.sqrt(-2 * np.log(us[2]) / us[2]) * us[0] * sigma + mu, 0
        )
        xs = xs[xs != 0]
        ys = np.where(
            us[2] <= 1, np.sqrt(-2 * np.log(us[2]) / us[2]) * us[1] * sigma + mu, 0
        )
        ys = ys[ys != 0]
        Zs = np.hstack((Zs, xs, ys))
        if len(Zs) >= n:
            break
    return Zs[:n]

def wiener_process(n):
    """
    Generuje n liczb z procesu Wienera o parametrach mu i sigma

    :param n: (int) liczba liczb do wygenerowania
    :param mu: (float) wartość oczekiwana
    :param sigma: (float) odchylenie standardowe
    :return: (np.ndarray) n liczb z procesu Wienera
    """
    h = 1 / n
    return np.concatenate((np.zeros(1), np.cumsum(norm_inv_cdf(n, 0, 1 / n))))


def ruin_probability(u, c, lamb, eta, T, N):
    r = 0
    for i in range(N):
        n = stats.poisson.rvs(mu=lamb * T)
        if n == 0:
            continue
        Us = T * np.random.uniform(size=n)
        Us = np.sort(Us)
        X = np.random.exponential(scale=eta, size=len(Us))
        R_t = u + c * Us - np.cumsum(X)

        if np.any(R_t < 0):
            r += 1
    return r / N


# us = np.linspace(0, 100, 100)
# ps = [ruin_probability(u, 2, 1, 1, 10000, 1000) for u in us]
# fig, ax = plt.subplots()
# ax.plot(us, ps)
# plt.show()


# ziggurat(2)





