# Import libraries
import pymc3 as pm
import numpy as np
import theano.tensor as tt

'''
How to work with MvNormal?
'''

with pm.Model() as model:
    cov = np.array([1., 0.5], [0.5, 2])
    mu = np.array([0, 0])

    obs = pm.MvNormal('obs', mu = mu, cov = cov, shape = (2, 2))
    print(tt.printing.Print('obs')(obs))
    
    '''
    Output: vals __str__ = [[0. 0.]
                           [0. 0.]]
    '''
