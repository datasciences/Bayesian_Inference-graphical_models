# Import libraries
import numpy as np
import pymc3 as pm


import numpy as np
import pymc3 as pm
import theano

x = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
x_shared = theano.shared(x)

with pm.Model() as model:
    p = pm.Beta('mu', 1, 1)
    obs = pm.Binomial('obs', n = 10, p = p, observed = x_shared)
    trace = pm.sample(1000)


pm.plot_trace(trace)


x_shared.set_value([0, 0, 0])
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=5)
