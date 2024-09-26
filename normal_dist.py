import numpy as np
import matplotlib.pyplot as plt
import timeit
import seaborn as sns
from scipy import stats, integrate, interpolate


def normal_box_muller(n, mu=0, sigma=1):
    U1, U2 = np.random.rand(n), np.random.rand(n)
    X = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    return mu + sigma * X


# def normal_polar(n, mu=0, sigma=1):
#     X = np.zeros(n)
#     i = 0
#     while i < n:  # pętla, znacznie gorzej działa niż wersja z Box-Mullerem
#         U1, U2 = 2 * np.random.rand() - 1, 2 * np.random.rand() - 1
#         R2 = U1 ** 2 + U2 ** 2
#         if R2 <= 1:
#             X[i] = mu + sigma * U1 * np.sqrt(-2 * np.log(R2) / R2)
#             i += 1
#     return X

def normal_polar(n, mu=0, sigma=1):
    X = np.zeros(n)
    # while True:
    U1, U2 = np.random.rand(n), np.random.rand(n)
    R2 = U1 ** 2 + U2 ** 2
    X = np.where(R2 <= 1, mu + sigma * U1 * np.sqrt(-2 * np.log(R2) / R2), 0)
    X = X[X != 0]
    return X


def posrednia():
    u1, u2, u3 = np.random.uniform(0,2, size=1000), np.random.uniform(0,2, size=1000), np.random.uniform(0,2, size=1000)
    # let x be middle value of 3 random variables
    xs = []
    for i in range(1000):
        xs.append(np.max([u1[i], u2[i], u3[i]]))

    f = lambda x: 3 / 8 * x ** 2 if 0 <= x <= 2 else 0

    sns.histplot(xs, stat='density', color='red', alpha=0.5)
    val = np.unique(xs)
    plt.plot(val, [f(x) for x in val], c='y', label='pdf theoretical')
    plt.show()


# normal distribution using exponential distribution and acceptance-rejection method
def normal_exp(n, mu=0, sigma=1):
    X = np.zeros(n)
    i = 0
    c = 1 / np.sqrt(2 * np.pi)
    while i < n:
        U1, U2 = np.random.rand(), np.random.rand()
        Y = -np.log(U1)
        if U2 <= np.exp(-0.5 * (Y - 1) ** 2):
        # if U2 <= 1/np.sqrt(2*np.pi) * np.exp(-0.5 * Y**2) / np.exp(-Y) / c:
            X[i] = mu + sigma * Y * np.sign(np.random.rand() - 0.5)
            i += 1
    return X




if __name__ == '__main__':
    # ns = np.arange(100, 10000, 100)
    # times_bm, times_p = [], []
    # for n in ns:
    #     times_bm.append(timeit.timeit(f'normal_box_muller({n})', globals=globals(), number=1))
    #     times_p.append(timeit.timeit(f'normal_polar({n})', globals=globals(), number=1))
    #
    # plt.plot(ns, times_bm, label='Box-Muller')
    # plt.plot(ns, times_p, label='Polar')
    # plt.legend()
    # plt.show()
    # print(len(normal_polar(1000)))
    # posrednia()

    sns.histplot(normal_exp(10000), stat='density', color='red', alpha=0.5)
    plt.plot(np.arange(-4, 4, 0.001), stats.norm.pdf(np.arange(-4, 4, 0.001)), c='y', label='pdf theoretical')
    plt.show()

    pass