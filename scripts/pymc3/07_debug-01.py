# Import libraries
import pymc3 as pm
import theano.tensor as tt

'''
Print intermediate variables in pymc3
https://docs.pymc.io/notebooks/howto_debugging.html
'''

# Model 1
with pm.Model() as model:
    mu = pm.Normal('mu', mu = 0, sigma = 2, shape = (2, 2))
    sd = pm.HalfNormal('sd', sigma = 1, shape = (2, 2))
    dc = pm.Dirichlet('dc', a = np.array([1.] * 3))

    mu_print = tt.printing.Print('mu')(mu)
    sd_print = tt.printing.Print('sd')(sd)
    dc_print = tt.printing.Print('dc')(dc)



# Model 2
'''
Concepts in Model 2 will come later. More details can be found here
https://docs.pymc.io/notebooks/LKJ.html
'''
def print_rv(a, b):
    tt.printing.Print(a)(b)
    
with pm.Model() as model:
    # Generalization of beta prior
    p = pm.Dirichlet('p', a=np.ones(4), shape = 4)  

    # Generalization of Binomial prior
    cc = pm.Categorical('c', p = p, shape = 2)    

    # prior for covariance matrix
    packed_L1 = pm.LKJCholeskyCov('packed_L1', 
                                 n = 2, 
                                 eta = 2.,
                                 sd_dist = pm.HalfCauchy.dist(6))

    # Convert to diagonal matrix
    L1 = pm.expand_packed_triangular(2, packed_L1)

    # Calculate covariance
    cov1 = pm.Deterministic('cov1', L1.dot(L1.T))

    # print distribution outputs
    print_rv('p', p)
    print_rv('cc', cc)
    print_rv('packed_L1', packed_L1)
    
'''

Model 1 Output:
mu __str__ = [[0. 0.]
 [0. 0.]]
sd __str__ = [[0.79788456 0.79788456]
 [0.79788456 0.79788456]]
dc __str__ = [0.33333333 0.33333333 0.33333333]


Model 2 output:
p __str__ = [0.25 0.25 0.25 0.25]
cc __str__ = [0 0]
packed_L1 __str__ = [1. 0. 1.]
'''
