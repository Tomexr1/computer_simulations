from random import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# zad 1
# orzeł 1, reszka 0

# a)
def g_won(tries=100):
    result = 0
    for _ in range(tries):
        seq = []
        while (seq[-3:] != [1,1,0]) and (seq[-3:] != [1,0,0]):
            if random() < 0.5:
                seq.append(1)
            else:
                seq.append(0)
        if seq[-3:] != [1,1,0]:
            result += 1
    return result / tries


# b)
def winner():
    for _ in range(1):
        seq = []
        while (seq[-3:] != [1, 1, 0]) and (seq[-3:] != [1, 0, 0]):
            if random() < 0.5:
                seq.append(1)
            else:
                seq.append(0)
        if seq[-3:] == [1, 1, 0]:
            return 'j'  # 2/3 szansy na wygranie
        else:
            return 'g'  # 1/3 szansy na wygranie


def zad1b(tries=100, g_start=25, j_start=5):
    # prawdopodbienstwo bankrutctwa grzesia
    result = 0
    for _ in range(tries):
        g = g_start
        j = j_start
        while g > 0 and j > 0:
            if winner() == 'g':
                j -= 1
                g += 1
            else:
                g -= 1
                j += 1
        if g <= 0:
            result += 1
    return result / tries


def zad1c():
    xs = np.arange(0, 20)
    ys_g, ys_j = [], []
    for x in xs:
        ys_g.append(zad1b(tries=1000, g_start=x, j_start=10))
        ys_j.append(1-(zad1b(tries=1000, g_start=10, j_start=x)))

    fig, ax = plt.subplots()
    ax.scatter(xs, ys_g, label='Grzesiek')
    ax.scatter(xs, ys_j, label='Jasiek')
    plt.title('Prawdopodobieństwo bankructwa gdy drugi gracz ma 10')
    plt.xlabel('Stan konta początkowego')
    plt.ylabel('Prawdopodobieństwo')
    ax.legend(loc='right')
    plt.show()


def demp(X, x):
    return len([i for i in X if i <= x]) / len(X)


def drawdemp(X):
    mx = max(X)+1
    mn = min(X)-1
    xs = np.arange(mn, mx, 0.001)
    plt.plot(xs, [demp(X, x) for x in xs])
    plt.show()


if __name__ == '__main__':
    # print(g_won(1))
    # print(zad1b(tries=10000))
    # print(zad1c())

    X = np.random.normal(0, 1, 1000)
    Y = np.random.uniform(0, 1, 1000)
    print(demp(Y, 1.2))
    drawdemp(X)
    print(demp(X, 0))
    print(demp(Y, 0))
    print(demp(Y, 1.2))