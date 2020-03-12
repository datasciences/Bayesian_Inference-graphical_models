# Import libraries
import numpy as np
import pymc3 as pm

'''
Assessing how a categorical distribution works with a normal distribution.

We generate a mixture of data points with 3 means.

Using a categorical distribution with proportions drawn from a dirichlet distribution
we create a categorical density function for each of the data points in our observation.

Further Using a normal distribution as likelihood, estimate the posterior

https://github.com/aloctavodia/Bayesian-Analysis-with-Python/blob/master/Chapter%207/07_Mixture_Models%20(1).ipynb

'''

import pymc3 as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

clusters = 3

n_cluster = [90, 50, 75]
n_total = sum(n_cluster)

means = [9, 21, 35]
std_devs = [2, 2, 2]

mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))


with pm.Model() as model_kg:
    # Each observation is assigned to a cluster/component with probability p
    p = pm.Dirichlet('p', a=np.ones(clusters))
    category = pm.Categorical('category', p=p, shape=n_total) 
    
    # Known Gaussians means
    means = pm.math.constant([10, 20, 35])

    y = pm.Normal('y', mu=means[category], sd=2, observed=mix)

    #### This step helps to understand the value of every data point and how it works with categorical distribution
    print(tt.printing.Print('y')(y))

    step1 = pm.ElemwiseCategorical(vars=[category], values=range(clusters))
    ## The CategoricalGibbsMetropolis is a recent addition to PyMC3
    ## I have not find the time yet to experiment with it.
    #step1 = pm.CategoricalGibbsMetropolis(vars=[category]) 
    step2 = pm.Metropolis(vars=[p])
    trace_kg = pm.sample(10000, step=[step1, step2])

    chain_kg = trace_kg[1000:]
    varnames_kg = ['p']
    pm.traceplot(chain_kg, varnames_kg)
    plt.savefig('B04958_07_03.png', dpi=300, figsize=[5.5, 5.5])
