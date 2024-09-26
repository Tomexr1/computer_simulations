from random import random
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import uniform
import timeit


def minimal_standard_lcg(a, c, m, n, seed=0):
    X = [seed]
    for i in range(1, n):
        X.append((a * X[i-1] + c) % m)
    return X


def uniform_lcg(n=1):
    return [i / 2e+32 for i in minimal_standard_lcg(69069, 5, 2e+32, n)]


def uniform_lcg_accuracy():
    xs = np.arange(-3, 3, 0.001)
    fig, axes = plt.subplots(2,1)
    axes[0].hist(uniform_lcg(100000), label='histogram z uniform_lcg')
    axes[1].plot(xs, uniform.pdf(xs), 'y-', label='pdf z uniform.pdf')
    fig.legend()
    plt.show()


def pi_lcg(n=1000):
    x, y = uniform_lcg(n), uniform_lcg(n)
    z = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return 4 * np.sum(z < 1) / n


def pi_corp(n=1000):
    x, y = van_der_corputt(2, n), van_der_corputt(3, n)
    z = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return 4 * np.sum(z < 1) / n


def pi_accuaracy():
    ns = range(1,1000)
    accuracies = []
    for n in ns:
        accuracies.append(abs(math.pi - pi_corp(n)))
    plt.scatter(ns, accuracies, s=1)
    plt.xscale='log'
    plt.xlabel('Liczba n prób')
    plt.ylabel('Dokładność')
    plt.show()


def van_der_corputt(b, n=1):
    xs = []
    for i in range(n):
        x = np.base_repr(i, b)[::-1]
        xs.append(int(x,b) / b ** len(x))
    return xs


def wykres_corputt():
    n = 50
    fig, ax = plt.subplots()
    ax.scatter(np.arange(-4,4,0.001), van_der_corputt(2, len(np.arange(-4,4,0.001))), s=1)
    # for i in range(2,10):
    #     ax.scatter(50*[i], van_der_corputt(i,n), s=1)
    # plt.xlabel('Baza')
    plt.ylabel('Zmienne')
    plt.show()


def pi_corp_plot():
    accuracies = []
    times = []
    ns = [100, 1000, 10000, 100000]
    for n in ns:
        times.append(timeit.timeit(f'pi_corp({n})',globals=globals(),number=1))
        accuracies.append(abs(math.pi - pi_corp(n)))

    plt.scatter(times, accuracies)
    plt.show()


def buffon_needle(l, d=1, n=1000):
    hits = 0
    for _ in range(n):
        x, phi = d /2 *uniform.rvs(), np.pi / 2 * random()  # x - odległość od linii, phi - kąt
        if x <= l / 2 * np.cos(phi):
            hits += 1
    return hits / n


def pi_buffon_needle(n=1000):
    l, d = 0.1, 1
    b_n = buffon_needle(l, d, n)
    if b_n:
        return l / d * 2 / b_n
    else:
        return 0


def pi_fromsquare(n=1000):
    # n has to be a square
    hits = 0
    xs, ys = [i / np.sqrt(n) for i in range(int(np.sqrt(n)))], [i / np.sqrt(n) for i in range(int(np.sqrt(n)))]
    for x in xs:
        for y in ys:
            if np.sqrt(x ** 2 + y ** 2) < 1:
                hits += 1
    return 4 * hits / n


def pi_comparison():
    # lcg vs Halton vs Buffon vs square; subplots for time(n), accuracy(n)
    ns_s = [i**2 for i in range(1,32)] # for square
    ns = np.arange(1, 1000, 1) # for lcg, halton, buffon
    times_lcg, times_halton, times_buffon, times_square = [], [], [], []
    accuracies_lcg, accuracies_halton, accuracies_buffon, accuracies_square = [], [], [], []
    for n in ns:
        times_lcg.append(timeit.timeit(f'pi_lcg({n})', globals=globals(), number=1))
        times_halton.append(timeit.timeit(f'pi_corp({n})', globals=globals(), number=1))
        times_buffon.append(timeit.timeit(f'pi_buffon_needle({n})', globals=globals(), number=1))
        accuracies_lcg.append(abs(math.pi - pi_lcg(n)))
        accuracies_halton.append(abs(math.pi - pi_corp(n)))
        accuracies_buffon.append(abs(math.pi - pi_buffon_needle(n)))
    for n in ns_s:
        times_square.append(timeit.timeit(f'pi_fromsquare({n})', globals=globals(), number=1))
        accuracies_square.append(abs(math.pi - pi_fromsquare(n)))
    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(ns, accuracies_lcg, label='lcg', s=1)
    axes[0].scatter(ns, accuracies_halton, label='halton', s=1)
    axes[0].scatter(ns, accuracies_buffon, label='buffon', s=1)
    axes[0].scatter(ns_s, accuracies_square, label='square', s=1)
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('accuracy')
    axes[0].legend()
    axes[1].scatter(ns, times_lcg, label='lcg', s=1)
    axes[1].scatter(ns, times_halton, label='halton', s=1)
    axes[1].scatter(ns, times_buffon, label='buffon', s=1)
    axes[1].scatter(ns_s, times_square, label='square', s=1)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('time')
    axes[1].legend()
    plt.show()


if __name__ == '__main__':
    # X = [i / 134456 for i in minimal_standard_lcg(8121, 28411, 134456, 1000)]
    # X = uniform_lcg(1000)
    # drawdemp(X)
    # uniform_lcg_accuracy()
    # print(pi(100000))
    # pi_accuaracy()
    # print(van_der_corputt(2, 10))
    # print(timeit.timeit('pi_corp(10000)',globals=globals(),number=1))
    # drawdemp(van_der_corputt(2, 30))
    # wykres_corputt()
    # print(pi_corp(10000))
    # pi_accuaracy()
    # pi_corp_plot()
    """Igła buffona do domu
    a) losowanie program
    b) zbieżność pi_n - pi
    c) porównanie z zad 3(szybkość, dokładność)
    """
    # print(buffon_needle(1, 1, 100000))
    # l, d = 1, 1
    # print(l / d * 2 / np.pi)
    # print(pi_buffon_needle(100000))
    # print(pi_fromsquare(10000))
    # print(pi_lcg(1000000))
    pi_comparison()