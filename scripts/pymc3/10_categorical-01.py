# Import libraries
import pymc3 as pm

'''
# https://discourse.pymc.io/t/pm-categorical-behaves-differently-in-a-model-versus-as-pm-categorical-dist/1675

'''
p = 1 / 3 * np.ones([30, 3])
with pm.Model() as model:
    p = T.as_tensor_variable(p)
    choice = pm.Categorical('choice', p=p)
    T.printing.Print('choice')(choice)  # prints choice __str__ = 0
    trace = pm.sample(100, tune=10, chains=1, cores=1)


with pm.Model() as model:
    p = pm.Uniform('p', lower=0, upper=1, shape=3)
    p_tile = T.tile(p, 30).reshape([30, 3])
    choice = pm.Categorical('choice', p=p_tile, observed=np.array([0, 0, 2] * 10))
    T.printing.Print('choice')(choice)  # prints choice __str__ = [0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0 2 0 0 2]
    trace = pm.sample(5000, tune=500, chains=2, cores=1)
print(pm.summary(trace))
