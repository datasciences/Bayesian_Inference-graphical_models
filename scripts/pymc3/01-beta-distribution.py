# -*- coding: utf-8 -*-
"""pymc3-init.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14iDQAUFMmlmn7A5RlwSv65Ubrd0u7NX3

# Beta distribution
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)

y = stats.beta(0.5, 0.5).pdf(x)

plt.plot(y)

"""# Beta distribution for several parameter values"""

params = [0.5, 1, 2, 3]

x = np.linspace(0, 1, 100)

f, ax = plt.subplots(len(params),
                     len(params),
                     sharex = True,
                     sharey = True)

for i in range(len(params)):
  for j in range(len(params)):
    y = stats.beta(params[i], params[j]).pdf(x)

    ax[i, j].plot(x, y)
    
ax[3, 0].set_xlabel('$\\theta$', fontsize = 14)
ax[0, 0].set_ylabel('$p(\\theta)$', fontsize = 14)

'''learnings:
1. The distribution is restricted to be between 0 and 1 and it can take
many different shapes.

2. The higher the value of alpha or beta, (because of
weight it will go down. pun intended!) therefore the
other side of the graph will go up

3. Beta distribution is the conjugate prior to binomial distribution.
Conjugate prior of a likelihood has the advantage that the posteior
distribution gives the same functional form as the prior

'''




# # The higher the value of alpha or beta, (because of weight it will go down. pun intended!) therefore the other side of the graph will go up"""# The higher the value of alpha or beta, (because of weight it will go down. pun intended!) therefore the other side of the graph will go up"""
