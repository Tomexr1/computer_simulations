import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def poisson_process(lamb, T):  # generuje momenty skoku
    n = stats.poisson.rvs(mu=lamb*T)
    if n == 0:
        return
    Us = T * np.random.uniform(size=n)
    Us = np.sort(Us)
    return Us


def risk_process(below_limit=3):
    T = 100
    lamb = 1
    ts = poisson_process(1, T)
    # loss_dist = stats.gamma(a=2, scale=1)
    loss_dist = stats.expon(scale=1)
    xs = loss_dist.rvs(size=len(ts))
    r0 = 10
    c = (1 + -0.15) * loss_dist.mean() * lamb
    p = lambda t: c * t
    t = 0
    dt = 1
    i = 0
    R = np.zeros(T+1)
    R[0] = r0
    below_counter = 0
    while below_counter <= below_limit and t < T:
        t += dt
        i += 1
        indx = len(ts[ts < t])
        R[i] = r0 + p(t) - np.sum(xs[:indx])
        if R[i] < 0:
            below_counter += 1
        else:
            below_counter = 0
    return R[:i+1]


if __name__ == '__main__':
    """ Proces ryzyka """
    pr = risk_process(below_limit=4)
    ts = np.arange(0, len(pr), 1)
    fig, ax = plt.subplots()
    ax.plot(ts, pr)
    ax.plot(ts, [0]*len(ts), color='red', linestyle='--')
    fig.set_size_inches(12, 6)
    plt.show()

    """ Rozkład czasu bankructwa """
    # N = 10000
    # bankruptcies = np.zeros(N)
    # for i in range(N):
    #     pr = risk_process(below_limit=0)
    #     bankruptcies[i] = len(pr)
    # fig, ax = plt.subplots()
    # sns.histplot(bankruptcies, stat='density', ax=ax, bins=20)
    # ax.set_title('Rozkład czasu bankructwa')
    # plt.show()
