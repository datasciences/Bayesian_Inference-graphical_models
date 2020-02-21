import numpy as np
import scipy.stats as st
import matplotlib.pylab as plt

n_exp = 10
X = []

for i in range(n_exp):
    outcome = st.bernoulli.rvs(p = 0.9, size = 5)

    while outcome.sum()!=5:
        outcome = np.append(outcome, st.bernoulli.rvs(p = 0.5))
    X.append(len(outcome))


plt.hist(np.asarray(X),
         np.arange(min(X), max(X)),
         rwidth = 0.75,
         density = True)
