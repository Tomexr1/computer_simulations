import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def mieszany_pc_pois1(dist=stats.norm, T=10):
    lamb = dist.rvs()
    t, I = 0, 0
    S = []
    while True:
        U = np.random.uniform()
        t -= np.log(U)/lamb
        if t > T:
            break
        I += 1
        S.append(t)
    return S


def mieszany_pc_pois2(dist=stats.norm, T=10):
    lamb = dist.rvs()
    n = stats.poisson(mu=T*lamb).rvs()
    if n == 0:
        return []
    Us = T * np.random.uniform(size=int(n))
    return sorted(Us)


def wiener_process(n):
    """
    Generuje n liczb z procesu Wienera o parametrach mu i sigma

    :param n: (int) liczba liczb do wygenerowania
    :param mu: (float) wartość oczekiwana
    :param sigma: (float) odchylenie standardowe
    :return: (np.ndarray) n liczb z procesu Wienera
    """
    return np.concatenate((np.zeros(1), np.cumsum(stats.norm.rvs(0, 1/n, size=n-1))))


def EB2():
    n = 1000
    EB2s = np.zeros(1000)

    for i in range(n):
        EB2s += wiener_process(1000) ** 2

    EB2s = EB2s / n
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 1, 0.001), EB2s, label='EB2(t)')
    ax.set_title('EB^2(t)')
    plt.show()


if __name__ == '__main__':
    """ ZAD 1 """
    # fig, axes = plt.subplots(2,2)
    # fig.set_size_inches(12, 6)
    # for i in range(10):
    #     ts_uniform = mieszany_pc_pois1(stats.uniform(scale=10))
    #     ts_exp = mieszany_pc_pois1(stats.expon())
    #     axes[0, 0].step(ts_uniform, np.cumsum(ts_uniform), where='post')
    #     axes[0, 1].step(ts_exp, np.cumsum(ts_exp), where='post')
    #     ts_uniform = mieszany_pc_pois2(stats.uniform(scale=10))
    #     ts_exp = mieszany_pc_pois2(stats.expon())
    #     axes[1, 0].step(ts_uniform, np.cumsum(ts_uniform), where='post')
    #     axes[1, 1].step(ts_exp, np.cumsum(ts_exp), where='post')
    # axes[0, 0].set_title('U(0,10), Metoda I')
    # axes[0, 1].set_title('Exp(1), Metoda I')
    # axes[1, 0].set_title('U(0,10), Metoda II')
    # axes[1, 1].set_title('Exp(1), Metoda II')
    # plt.show()

    """ ZAD 2 """
    # fig, axes = plt.subplots(3, 2, layout='tight')
    # fig.set_size_inches(12, 6)
    # rvs = [stats.uniform(scale=10), stats.expon(), stats.gamma(a=2)]
    # for j in [0,1,2]:
    #     N_half_period = np.zeros(500)
    #     N_full_period = np.zeros(500)
    #     for i in range(500):
    #         ns = np.cumsum(mieszany_pc_pois1(rvs[j]))
    #         if len(ns) > 0:
    #             N_half_period[i] = ns[len(ns)//2]
    #             N_full_period[i] = ns[-1]
    #     sns.histplot(N_full_period, stat='density', ax=axes[j, 0])
    #     sns.histplot(N_half_period, stat='density', ax=axes[j, 1])
    #     axes[j, 0].set_title('N_full_period')
    #     axes[j, 1].set_title('N_half_period')
    # plt.show()

    """ ZAD 3 """
    """ Jednowymiarowy """
    # h = 0.001
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(0, 1, h), wiener_process(1000), label='W_t')
    # ax.set_title('Przykładowa trajektoria realizacji procesu Wienera')
    # plt.show()

    """ Dwuwymiarowy """
    h = 0.001
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(projection='3d')
    X, Y = wiener_process(1000), wiener_process(1000)
    X, Y = np.meshgrid(X, Y)
    ax.scatter(wiener_process(1000), wiener_process(1000), wiener_process(1000), label='W_t')
    ax.set_title('Przykładowa trajektoria realizacji dwuwymiarowego procesu Wienera')
    plt.show()

    """ Wykres EB^2(t)"""
    # EB2()

