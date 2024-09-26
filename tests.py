import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scipy.interpolate as interp

gamma = 2
delta = 3

X = gamma / (stats.norm.rvs(size=10000) ** 2) + delta
# sns.ecdfplot(Ys)
# sns.lineplot(x=np.unique(Ys), y=2 - 2*stats.norm.cdf(np.sqrt(gamma/ (np.unique(Ys) - delta)),
#                                                      loc=0, scale=1), color='red')
# sns.histplot(Ys, stat='density')
# sns.lineplot(x=np.unique(Ys), y=np.sqrt(gamma/2/np.pi) * np.exp(-gamma/2/(np.unique(Ys) - delta)) / (np.unique(Ys) - delta) ** 1.5, color='red')

# test if Ys are from levy distribution with gamma=2, delta=3
stats.probplot(X, dist='levy', sparams=(gamma, delta), plot=plt)

plt.show()