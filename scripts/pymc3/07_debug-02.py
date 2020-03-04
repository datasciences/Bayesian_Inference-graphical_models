# Import libraries
import pymc3 as pm
import theano.tensor as tt

'''
Print intermediate variables in pymc3
https://docs.pymc.io/notebooks/howto_debugging.html
'''

# Generate data
x = np.random.randn(100)

# Model
with pm.Model() as model:
    mu = pm.Normal('mu', mu = 0, sigma = 1)
    sd = pm.HalfNormal('sd', sigma = 1)

    mu_print = tt.printing.Print('mu')(mu)
    sd_print = tt.printing.Print('sd')(sd)

    obs = pm.Normal('obs', mu = mu_print, sigma = sd_print, observed = x)

    trace = pm.sample(110, tune = 0, chains = 1, progressbar = True)
