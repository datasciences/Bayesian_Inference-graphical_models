'''
A simple normal mixture model
'''

# Import libraries
import pymc3 as pm
import numpy as np


# create data
sample = np.hstack([np.random.randn(100),np.random.rand(100)])

# Model
with pm.Model() as m:

    # Define parameters as random variables with prior
    mu = pm.Normal('mu')
    sd = pm.HalfNormal('sd', 1)
    condition = pm.Normal.dist(mu, sd)
    rn = pm.Uniform.dist(-5., 5.)

    # likelihood as mixture distribution
    sampling_dist = pm.Mixture(
        "Sampling distribution ",
        w=[0.5, 0.5],
        comp_dists=[condition, rn],
        observed=sample)

# sample using NUTS
trace = pm.sample(1000)
az.plot_trace(trace)


# Sample from a normal mixture
with pm.Model() as model:
    mus = pm.Normal('mus', shape=(6,12))
    taus = pm.Gamma('taus', alpha=1, beta=1, shape=(6, 12))
    ws = pm.Dirichlet('ws', np.ones(12))
    mixture = pm.NormalMixture('m', w=ws, mu=mus, tau=taus, shape=6)
