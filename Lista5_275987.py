import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def poisson_process(lamb, T):  # generuje czasy oczekiwania na zdarzenia, zwraca czasy skoków
    n = 0
    t = 0
    ts = []
    while True:
        u = np.random.random()
        t += -1 / lamb * np.log(u)
        if t > T:
            break
        ts.append(t)
        n += 1
    return np.array(ts)


def trajectory(lamb, T, process):
    lambdas = [lamb, lamb/2, lamb*2]
    fig, ax = plt.subplots()
    for lamb in lambdas:
        ts = process(lamb, T).tolist()
        ys = np.arange(0, len(ts)+1, 1)
        ax.step([0]+ts, ys, where='post')
        ax.set_xlim(-1, T+1)
    plt.show()


def poisson_test(lamb, T, process=poisson_process):
    Ns = np.zeros(100)
    for _ in range(100):
        ts = process(lamb, T)
        Ns[_] = len(ts)
    stats.probplot(Ns, dist='poisson', sparams=(lamb*T,), plot=plt)
    plt.show()


def compensated_poisson_process(lamb, T):
    ts = poisson_process(lamb, T)
    ys = np.arange(1, len(ts)+1, 1) - lamb * np.array([0] + ts)
    fig, ax = plt.subplots()
    ts.tolist()
    ys.tolist()
    ax.step([0]+ts, [0]+ys, where='post')
    ax.set_xlim(-1, T+1)
    plt.show()


def poisson_process2(lamb, T):  # generuje momenty skoku
    n = stats.poisson.rvs(mu=lamb*T)
    if n == 0:
        return
    Us = T * np.random.uniform(size=n)
    Us = np.sort(Us)
    return Us


def pois1(lamb=1, n=1):  # dla niewielkich lambd
    Xs = np.zeros(n)
    for i in range(n):
        n, a = 1, 1
        while True:
            U = np.random.uniform()
            a *= U
            if a < np.exp(-lamb):
                Xs[i] = n - 1
                break
            n += 1
    return Xs


def pois2(lamb=1, n=1):
    Xs = np.zeros(n)
    for i in range(n):
        m = np.floor(7*lamb/8)
        Y = stats.gamma.rvs(a=m, scale=1)
        if Y <= lamb:
            Z = pois1(lamb=lamb-Y)
            Xs[i] = m + Z
        else:
            Xs[i] = stats.binom.rvs(n=int(m-1), p=lamb/Y)
    return Xs


def pois_gen_test(gen=pois1, lamb=1):
    X = gen(lamb, 1000)
    fig, ax = plt.subplots(3, 1, layout="constrained")
    val = np.unique(X)
    ax[0].plot(
        val,
        stats.poisson.pmf(val, mu=lamb),
        c="y",
        label="pmf theoretical",
    )
    sns.histplot(
        X, stat="probability", ax=ax[0], color="red", alpha=0.5, label="pmf empirical", binwidth=1
    )
    ax[1].plot(
        val,
        stats.poisson.cdf(val, mu=lamb),
        label="cdf theoretical",
        linestyle=(0, (5, 10)),
    )
    sns.ecdfplot(X, label="cdf empirical", color="red", alpha=0.5, ax=ax[1])
    stats.probplot(X, dist="poisson", sparams=(lamb,), plot=ax[2])
    fig.legend()
    plt.show()


if __name__ == '__main__':
    # trajectory(3, 10, process=poisson_process)  # 3 traktorie procesu Poissona metoda 1
    # trajectory(3, 10, process=poisson_process2)  # 3 traktorie procesu Poissona metoda 2
    # poisson_test(3, 10, process=poisson_process)  # test czy N(T) ma rozkład Poissona dla metody 1
    poisson_test(3, 10, process=poisson_process2)  # test czy N(T) ma rozkład Poissona dla metody 2
    # compensated_poisson_process(3, 100)  # proces Poissona z kompensacją
    # pois_gen_test(pois1, 2)  # testowanie rozkładu Poissona metodą 1
    # pois_gen_test(pois2, 400)  # testowanie rozkładu Poissona metodą 2

    pass
