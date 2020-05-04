# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 15:34:03 2020

@author: owner
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.neighbors import KernelDensity

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

plt.clf()

x = make_data(1000)
x_d = np.linspace(-4, 8, 100)

hist = plt.hist(x, bins=30, density=True)

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=0.3, kernel='gaussian')
kde.fit(x[:, None])

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d[:, None])
prob = np.exp(logprob)
plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)


plt.tight_layout()