import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import interpolate


def inh_poisson_process(lamb, T):
    M = np.max([lamb(t) for t in np.arange(0, T, 0.01)])
    t, I = 0, 0
    ts = []
    while True:
        Us = np.random.uniform(size=2)
        t -= np.log(Us[0]) / M
        if t > T:
            break
        if Us[1] <= lamb(t)/M:
            I += 1
            ts.append(t)
    return ts


def inh_poisson_process2(lamb, T):
    m = lambda t: integrate.quad(lamb, 0, t)[0]
    m_inv = interpolate.interp1d([m(t) for t in np.arange(0, T, 0.01)], np.arange(0, T, 0.01))
    T_tilde = m(T)
    t, I = 0, 0
    ts = []
    while True:
        U = np.random.uniform()
        t -= np.log(U)
        if t > T_tilde:
            break
        I += 1
        ts.append(m_inv(t))
    return ts


def poisson_test(lamb, T, process=inh_poisson_process):
    Ns = np.zeros(100)
    for _ in range(100):
        ts = process(lamb, T)
        Ns[_] = len(ts)
    stats.probplot(Ns, dist='poisson', sparams=(integrate.quad(lamb, 0, T)[0],), plot=plt)
    plt.show()


def join_posisson_process(lmbds, T):
    process1 = inh_poisson_process(lmbds[0], T)
    process2 = inh_poisson_process(lmbds[1], T)
    times = sorted(np.hstack((process1, process2)))
    return times


def join_poisson_process_plot(lmbds, T):
    process1 = inh_poisson_process(lmbds[0], T)
    process2 = inh_poisson_process(lmbds[1], T)
    times = sorted(np.hstack((process1, process2)))

    fig, ax = plt.subplots()
    ax.step([0] + list(process1), [0] + list(np.arange(1, len(process1)+1)), where='post')
    ax.step([0] + list(process2), [0] + list(np.arange(1, len(process2)+1)), where='post')
    ax.step([0] + list(times), [0] + list(np.arange(1, len(times)+1)), where='post')
    ax.set_xlim(-1, T + 1)
    plt.show()


def join_poisson_process_test(lmbds, T):
    Ns = np.zeros(100)
    for i in range(100):
        Ns[i] = len(join_posisson_process(lmbds, T))

    stats.probplot(Ns, dist='poisson', sparams=(integrate.quad(lambda t: lmbds[0](t)+lmbds[1](t), 0, T)[0],), plot=plt)
    plt.show()


def compound_poisson_process_traj(lamb, T):
    fig, ax = plt.subplots()
    times = inh_poisson_process(lamb, T)
    Ns = np.cumsum(stats.norm.rvs(size=len(times)))
    ax.step([0]+list(times), [0]+list(Ns), where='post')
    ax.set_xlim(-1, T + 1)
    plt.show()


def trajectory(lamb, T, process):
    fig, ax = plt.subplots()
    ts = process(lamb, T)
    ys = np.arange(0, len(ts)+1, 1)
    ax.step([0]+ts, ys, where='post')
    ax.set_xlim(-1, T+1)
    plt.show()


if __name__ == '__main__':
    lambdas = [lambda t: (np.sin(t)) ** 2,
               lambda t: t ** 4,
               lambda t: np.exp(-t ** 2),
               lambda t: t,
               lambda t, p: np.sign(np.sin(p * t)) + 1,
               lambda t, p: min((264, np.abs(t - p / 3 - 1 / 3))),
               lambda t: np.exp((2 * np.sin(t / 12 + 3) - 0.9) ** 3)]
    """Całkowanie lambda(t) od 0 do T"""
    # T = 5
    # for l in lambdas[:4]:
    #     print(integrate.quad(l, 0, T))
    # for l in lambdas[4:6]:
    #     print(integrate.quad(l, 0, T, args=(2,)))
    # print(integrate.quad(lambdas[6], 0, T))

    """ZAD 1 - przerzedzanie"""
    # poisson_test(lambdas[0], 100, inh_poisson_process)  # test rozkladu intensywnosci w t=T
    trajectory(lambdas[0], 100, inh_poisson_process)  # trajektoria

    """ZAD 2 - N(t) = N_tilde(quad(lambda(t), 0 , t)), N_tilde - jednorodny proces Poissona"""
    # poisson_test(lambdas[6], 10, inh_poisson_process2)  # test rozkladu intensywnosci w t=T
    # trajectory(lambdas[6], 100, inh_poisson_process2)  # trajektoria

    """ZAD 3 - łączenie procesów Poissona"""
    # join_poisson_process_test((lambdas[0], lambdas[6]), 100)  # test rozkladu intensywnosci intensywności w t=T
    # join_poisson_process_plot((lambdas[0], lambdas[6]), 100)  # trajektorie

    """Zad 4 trajektoria"""
    # compound_poisson_process_traj(lambdas[0], 100)  # trajektoria, zmienne Z_i generowane z N(0,1)