'''
A script to show how distributions can be defined
outside of pymc3 context manager.

Ref: Pymc3 developer guide

Distributions cannot be defined outside of the
context manager. This is because distribution clases
are designed to integrate themselves automatically inside
of a pymc3 model. Each time a random variable is defined
they are added to the pymc3 model.

Every distribution has two methods "random" and logp
random- draws a random sample from the distribution
logp - gives the log probability of the value

'''

y = pm.Bernoulli.dist(p = 0.9)

# calculate logprobability of 4
y.logp(4).eval()

# draw random numbers from the distribution
y.random(size = 3)

# Look at the points drawn from a bernouli distribution
# Since the probability of 1 is 0.9, you will see a plot with high density at 1
sns.distplot(y.random(size = 100))


# for LKJ corr
y = pm.LKJCorr.dist(eta = 1, n = 2)

# Look at the points drawn from a bernouli distribution
sns.distplot(y.random(size = 100))
