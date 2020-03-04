import pymc3 as pm
import numpy as np

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

# Simple normal model
with pm.Model() as model:
    '''
    mu is a random variable and mu.dist is a density function(or distribution)
    Since it is a function, you can evaluate any value using that function.
    '''
    mu = pm.Normal('mu', mu = 0, sd = 1)
    obs = pm.Normal('obs', mu = mu, sd = 1,
                    observed = np.random.randn(100))

    # print variables 
    print(model.basic_RVs, model.free_RVs, model.observed_RVs)

    # print logp of model
    print(model.logp({'mu': 0}))

    # print logp of mu
    print(mu.logp({'mu':0})
    
# LKJ prior
with pm.Model() as model:
    packed_L = [pm.LKJCholeskyCov('packed_L_%d' % i, n=2,
                                  eta=2.,
                                  sd_dist=pm.HalfCauchy.dist(1)) for i in range(4)]
    print(model.basic_RVs)
    print(model.free_RVs)
    print(model.observed_RVs)
