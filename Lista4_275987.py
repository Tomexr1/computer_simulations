import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats


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


def box_muller_polar(n=1, mu=0, sigma=1):
    """
    Generuje n liczb z rozkładu normalnego o parametrach mu i sigma za pomocą metody biegunowej

    :param n:
    :param mu:
    :param sigma:
    :return:
    """
    Zs = []
    while True:
        us = np.array(
            [
                np.random.uniform(-1, 1, size=int(np.ceil(0.6 * (n - len(Zs))))),
                np.random.uniform(-1, 1, size=int(np.ceil(0.6 * (n - len(Zs))))),
            ]
        )  # expected value powtórzeń ~ 1.2
        R_sq = np.sum(us**2, axis=0)
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


def plots_stats(gen=box_muller):
    X = gen(1000, 1, 2)
    print("Expected value: empirical:", np.average(X), "theoretical:", 1)
    print("Variance: empirical:", np.sqrt(np.var(X)), "theoretical:", 2)
    fig, ax = plt.subplots(3, 1, layout="constrained")
    val = np.unique(X)
    ax[0].plot(
        val,
        [stats.norm.pdf(v, loc=1, scale=2) for v in val],
        c="y",
        label="pdf theoretical",
    )
    sns.histplot(
        X, stat="density", ax=ax[0], color="red", alpha=0.5, label="pdf empirical"
    )
    ax[1].plot(
        val,
        [stats.norm.cdf(v, loc=1, scale=2) for v in val],
        label="cdf theoretical",
        linestyle=(0, (5, 10)),
    )
    sns.ecdfplot(X, label="cdf empirical", color="red", alpha=0.5, ax=ax[1])
    stats.probplot(X, dist="norm", sparams=(1, 2), plot=ax[2])
    fig.legend()
    plt.show()


def times_comparison():
    import time

    ns = np.arange(100, 100000, 100)
    bm, bmp = [], []
    for n in ns:
        start = time.time()
        box_muller(n, 1, 2)
        bm.append(time.time() - start)
        start = time.time()
        box_muller_polar(n, 1, 2)
        bmp.append(time.time() - start)
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(ns, bm, label="Box-Muller")
    ax.plot(ns, bmp, label="Box-Muller Polar")
    ax.set_xlabel("n")
    ax.set_ylabel("time")
    ax.legend()
    plt.show()


def chol_decomp(A):
    if not np.all(np.linalg.eigvals(A) > 0):
        raise ValueError("Matrix is not positive definite")
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric")
    n = A.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :] ** 2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :] * L[j, :])) / L[j, j]
    return L


def normND(
    size=1,
    sigma: np.ndarray = np.array([[1, 0], [0, 1]]),
    mu: np.ndarray = np.array([0, 0]),
):
    Xs = np.zeros((sigma.shape[0], size))

    # L = np.linalg.cholesky(sigma)
    # Z = box_muller(size*sigma.shape[0], 0, 1)
    # Xs = Xs[:, mu + L @ np.array_split(Z, sigma.shape[0], axis=)]

    for i in range(size):
        Z = box_muller(sigma.shape[0], 0, 1)
        L = np.linalg.cholesky(sigma)
        Xs[:, i] = mu + L @ Z
    return Xs


def chol_plot():
    Xs = normND(10000, np.array([[1, -0.99], [-0.99, 1]]), np.array([0, 0]))
    sns.jointplot(x=Xs[0, :], y=Xs[1, :], kind="kde", fill=True)
    plt.show()


def chol_plot_3d():
    Xs = normND(10000, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Xs[0, :], Xs[1, :], Xs[2, :], c="r", marker="o", s=1)
    plt.show()


if __name__ == "__main__":
    # plots_stats(gen=box_muller)  # sprawdzenie poprawności box_muller
    # plots_stats(gen=box_muller_polar)  # sprawdzenie poprawności box_muller_polar
    # times_comparison()  # porównanie czasów wykonania obu metod
    # chol_plot()  # wykres 2D
    chol_plot_3d()  # wykres 3D

    pass
