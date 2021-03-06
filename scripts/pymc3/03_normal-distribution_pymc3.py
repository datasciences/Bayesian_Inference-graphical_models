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

    # it takes the free parameters mu and sigma as input and output
    # the sum of logp of the observations conditioned on the input.    
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


Side notes
------
The MCMC sampler draws parameter values from the prior distribution
and computes the likelihood that the observed data came from
a distribution with these parameter values.

We observe counts of data (y) for each conversation i (Observed Data)
This data was generated by a random process which we believe can be
represented using a particular distribution (Likelihood)
This distribution has a few parameters that we define using mu sigma etc

the MCMC sampler wanders towards areas of highest likelihood.
However, the Bayesian method is not concerned with findings
the absolute maximum values - but rather to traverse and
collect samples around the area of highest probability.
All of the samples collected are considered to be a credible parameter.

'''
