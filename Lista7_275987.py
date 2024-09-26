import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import interpolate
from raport import ruin_probability


def ecf(X, x):
    return np.average(np.exp(1j * x * X))


def poisson_process2(lamb, T):  # generuje momenty skoku
    n = stats.poisson.rvs(mu=lamb*T)
    if n == 0:
        return
    Us = T * np.random.uniform(size=n)
    Us = np.sort(Us)
    return Us


def compound_poisson_process_traj(lamb, T, dist=stats.norm):
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(12, 6)
    times = poisson_process2(lamb, T)
    Ns = np.cumsum(dist.rvs(size=len(times)))
    Ns_T = np.zeros(10000)
    for i in range(10000):
        ts = poisson_process2(lamb, T)
        Ns_T[i] = np.cumsum(dist.rvs(size=len(ts)))[-1]

    ax[0].set_title('Compound Poisson Process Trajectory')
    ax[0].step([0]+list(times), [0]+list(Ns), where='post', label='Trajectory')
    ax[0].set_xlim(-1, T + 1)

    if dist == stats.norm:
        cf = lambda u: np.exp(lamb * T * (np.exp(-0.5 * (u**2))-1))
    else:  # dist == stats.cauchy
        cf = lambda u: np.exp(lamb * T * (np.exp(-np.abs(u))-1))

    ax[1].set_title('Characteristic Function t = T')
    domain = np.linspace(-6, 6, 1000)
    ax[1].plot(domain, np.abs(cf(domain)),
               label='abs(Characteristic Function) - theoretical', color='red')
    ax[1].plot(domain, np.abs([ecf(Ns_T, u) for u in domain]),
               label='abs(Characteristic Function) - empirical', color='blue', linestyle='--')

    fig.legend(draggable=True)
    plt.show()


def risk_process():
    lamb = 1
    loss_dist = stats.expon(scale=1)
    r0 = 5
    c = 0.5
    p = lambda t: c * t
    t = 0
    i = 0
    R = [r0]
    while t != 10:
        t += 1
        i += 1
        s = poisson_process2(lamb, t)
        if s is not None:
            R.append(r0 + p(t) - np.sum(loss_dist.rvs(size=len(s))))
        else:
            R.append(r0 + p(t))
    return R


def risk_process_probabilities():
    for t in [1, 3, 5]:
        print(f'P(R_{t} < 0) = {ruin_probability(5, 0.5, 1, 1, t, 10000)}')


if __name__ == '__main__':
    """Zad 1/2"""
    # compound_poisson_process_traj(2, 10, stats.cauchy)

    """Zad 3"""
    risk_process_probabilities()