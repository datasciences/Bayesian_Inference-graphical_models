import pymc3 as pm
from scipy.stats import bernoulli
'''
http://alfan-farizki.blogspot.com/2015/07/pymc-tutorial-bayesian-parameter.html
'''

def generate_sample(t, s):
    return bernoulli.rvs(t, size = s)    
  
def model(data):
    with pm.Model() as model:
        theta_prior = pm.Beta('theta_prior', alpha = 1.0, beta = 1.0)
        coin = pm.Bernoulli('coin', p = theta_prior, observed = data)

    return model
  
def generate_traces(model):
    '''MCMC will perform several iterations to generate
    the sample from “theta_prior”, in which each iteration
    will improve the quality of the sample.'''
    
    with model:
        trace = pm.sample(100)
    return model, trace

def pymc3_plot(model, trace):
    with model:
        pm.plot_trace(trace)
        
def custom_plot(trace):
    plt.hist(trace['theta_prior'])

data = generate_sample(0.2, 50)
model = model(data)
model, trace = generate_traces(model)
pymc3_plot(model, trace)
custom_plot(trace)
