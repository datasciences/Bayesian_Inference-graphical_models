'''
https://stats.stackexchange.com/questions/152020/using-pymc-to-solve-the-simple-cancer-problem

http://yudkowsky.net/rational/bayes
'''


#### simple_cancer.py
import pylab as pl
import pymc as mc

# 1% of women at age forty who participate in routine screening have
# breast cancer.  80% of women with breast cancer will get positive
# mammographies.  9.6% of women without breast cancer will also get
# positive mammographies.  A woman in this age group had a positive
# mammography in a routine screening.  What is the probability that
# she actually has breast cancer?

POS_obs = [1.]
N = len(POS_obs)

C = mc.Bernoulli('C', .01)

@mc.deterministic
def p_POS(C=C):
    if C:
        return .8
    else:
        return .096

POS = mc.Bernoulli('POS', p_POS, value=POS_obs, observed=True)

### run_simple_cancer.py
import pymc as mc
import simple_cancer
m = mc.MCMC(simple_cancer)
m.sample(100000)

c = m.trace('C')[:]
print sum(c), 1.0*sum(c)/len(c)
