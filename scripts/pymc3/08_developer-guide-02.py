import pymc3 as pm
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

'''
Understand the difference between basic, free and observed random variables.

In general, if a variable has observations (observed parameter), the RV is
defined as an ObservedRV, otherwise if it has a transformed (transform
parameter) attribute, it is a TransformedRV, otherwise, it will be the
most elementary form: a FreeRV. Note that this means that random
variables with observations cannot be transformed.

https://docs.pymc.io/developer_guide.html
https://docs.pymc.io/notebooks/api_quickstart.html
'''

with pm.Model() as model:

    # A normal model
    mu = pm.Normal('mu', mu = 0, sd = 1)
    obs = pm.Normal('mu', mu = mu, sd = 1,
                    observed = np.random.randn(100))

    # logp of model and mu
    print(model.logp({'mu':0})
    print(mu.logp({'mu':0}))

    # to store logp of the model 
    llh = pm.Deterministic('llh', model.logpt)
    trace = pm.sample(300, chains = 4)
    
    print(model.basic_RVs, model.free_RVs, model.observed_RVs)
    
pm.plot_trace(trace)

logp = model.logp
lnp = np.array([logp(trace.point(i, chain = c))for c in trace.chains for i in range(len(trace)])
plt.plot(lnp)

trace.point(0, 1) # sample no and chain no


# Plot logp values for a binomial distribution
y = pm.Binomial.dist(n=10, p=0.5)
plt.plot([y.logp(i).eval() for i in range(11)])




          
