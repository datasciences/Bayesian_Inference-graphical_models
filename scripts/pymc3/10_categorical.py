# Import libraries
import numpy as np
import pymc3 as pm

'''
# https://discourse.pymc.io/t/pm-categorical-behaves-differently-in-a-model-versus-as-pm-categorical-dist/1675

Sampling from a Categorical distribution:
1. Generate an array of 1000 samples drawn according to a defined
distribution.
2. Using a dirichlet distribution define alpha values being uniform.
3. Model likelihood as a categorical distribution.
4. Draw samples from dirichlet distribution to get posterior distribution
of the numbers.

'''

# -----Experiment 1
preal = [0.1, 0.1, 0.2, 0.6]
y = np.random.choice(4, 1000, p=preal)

with pm.Model():
    
    p = pm.Dirichlet('p', a=np.ones(4))

    pm.Categorical(name='y', p=p, observed=y)

    trace = pm.sample(draws=2000, tune=200)

#pm.traceplot(trace, varnames=['p'], lines=dict(p=preal))
pm.plot_trace(trace)
pm.plot_posterior(trace)


# ---- Experiment 2
'''
Dirichlet distributio of 2 categories of 2 variables
'''
with pm.Model() as model:    
      p = pm.Dirichlet('p', a = np.ones(2), shape = (2, 2))
      print(tt.printing.Print('p')(p))
