# -*- coding: utf-8 -*-
"""
Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1I1bpOj-36lMYcMh857eeYaYh959Iu5wk
"""

# Commented out IPython magic to ensure Python compatibility.
import pymc3 as pm
import numpy as np
import seaborn as sns

# create random variable which is normally distributed
y = np.random.normal(10, 4, 1000)

# Examine the density plot of y
sns.kdeplot(y)

# create a pymc3 model
with pm.Model() as model:

    #  an uninformative uniform prior between 0 and 100
    mu = pm.Uniform('mu', 0, 100)

    # An uninformative uniform prior inbetween 0 and 4
    sigma = pm.Uniform('sigma', 0, 4)
    
    likelihood = pm.Normal('likelihood', mu = mu, sigma = sigma, observed = y)

    trace = pm.sample(1000)

    pm.traceplot(trace)

    pm.plot_posterior(trace)




'''
You should play with the alpha and beta parameters of
beta distribution and see how the HPD changes.

You should also play with n and observed, meaning increase the
number of data points.

When you use a strong prior and little data, you will see
the posterior resemble the shape of the prior. Whereas for the same prior,
if you increase the number of datapoints, you will see influence of data more strong

'''
