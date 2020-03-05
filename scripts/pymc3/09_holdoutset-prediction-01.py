# Import libraries
import pymc3 as pm
import theano

'''
Making predictions in pymc3
https://docs.pymc.io/notebooks/api_quickstart.html

You need to use the shared functionality. You use the same
computational graph created by theano, only add the test data
and make predictions.
'''

x = np.random.randn(100)
y = x > 0

x_shared = theano.shared(x)
y_shared = theano.shared(y)

with pm.Model() as model:
    coeff = pm.Normal('x', mu = 0, sigma = 1)
    logistic = pm.math.sigmoid(coeff * x_shared)
    pm.Bernoulli('obs', p = logistic, observed = y_shared)

    trace = pm.sample()

x_shared.set_value([-1, 0, 1.])
y_shared.set_value([0, 0, 0])

with model:
    post_pred = pm.sample_posterior_predictive(trace, samples= 500)

post_pred['obs'].mean(axis=0)
