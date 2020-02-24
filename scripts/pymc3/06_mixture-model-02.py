'''
Mixture model from Bayesian data analysis- Oswaldo martin

Seaborn dataset 'tips' is used for the analysis. It has daily tips data in several hotels
on Thursday, Friday, Saturday and Sunday.

A gaussian mixture model should be build to check the distribution of
tips on working days. 
'''

import seaborn as sns
import pymc3 as pm
import pandas as pd

# load and plot tips on working days
tips = sns.load_dataset('tips')
tips.tail()

sns.violinplot(x = 'day', y = 'tip', data = tips)

# get tip and days data
y = tips['tip'].values
idx = pd.Categorical(tips['day']).codes

# build model
with pm.Model() as model:

    mu = pm.Normal('mu', mu= 0, sd = 10, shape = len(set(idx)))
    sigma = pm.HalfNormal('sigma', sd = 10, shape = len(set(idx)))
    output = pm.Normal('output', mu = mu[idx], sd = sigma[idx], observed = y)

    trace = pm.sample(2000)

    pm.plot_trace(trace)

    pm.plot_posterior(trace)

    pm.gelman_rubin(trace)
