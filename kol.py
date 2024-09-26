import numpy as np
import scipy.stats as stats

def risk_process(below_limit=4):
    T = 100
    lamb = 1
    ts = inh_poisson_process(lambda t: 1 + np.sin(t), T)
    # loss_dist = stats.gamma(a=2, scale=1)
    loss_dist = stats.expon(scale=1/A)
    xs = loss_dist.rvs(sizae=len(ts))
    r0 = 2 * A
    c = A / 3
    p = lambda t: c * t
    t = 0
    dt = 1
    i = 0
    R = np.zeros(T+1)
    R[0] = r0
    below_counter = 0
    while below_counter <= below_limit and t < T and R[i] > -10:
        t += dt
        i += 1
        indx = len(ts[ts < t])
        R[i] = r0 + p(t) - np.sum(xs[:indx])
        if R[i] < 0:
            below_counter += 1
        else:
            below_counter = 0
    return R[:i+1]


# prawodpodobieństwo bankructwa w ciągu 5 lat
N = 10000
bankruptcies = np.zeros(N)
for i in range(N):
    pr = risk_process(below_limit=0)
    bankruptcies[i] = len(pr)

print(np.mean(bankruptcies <= 5))
