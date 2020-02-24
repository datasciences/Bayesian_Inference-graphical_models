'''
In this tutorial we start with a StudentT distribution and
Half normal distribution to model the likelihood and the standard deviation

'''

import numpy as np
import seaborn as sns
import pymc3 as pm

def main():
    
    data = np.array([51.06, 55.12, 53.73, 50.24, 52.05, 56.40, 48.45, 52.34, 55.65, 51.49, 51.86, 63.43, 53.00, 56.09, 51.93, 52.31, 52.33, 57.48, 57.44, 55.14, 53.93, 54.62, 56.09, 68.58, 51.36, 55.47, 50.73,
                     51.94, 54.95, 50.39, 52.91, 51.5, 52.68, 47.72, 49.73, 51.82, 54.99, 52.84, 53.19, 54.52, 51.46, 53.73, 51.61, 49.81, 52.42, 54.3, 53.84, 53.16])

    # look at the distribution of the data
    sns.kdeplot(data)

    # All these distributions are used to model std
    # It is safe to use exponential
    # half cauchy has a fat tail
    # Exponential parameter lambda high indicates a high steep
    # Ineverse gamma 
    with pm.Model() as model:
      mu = pm.Uniform('mu', 30, 80)
      sigma = pm.HalfNormal('sigma', sd = 10)
      df = pm.Exponential('df', 1.5) # lamda = 1.5, it will be more steep, 0.5 less
      output = pm.StudentT('output',
                       mu =  mu,
                       sigma = sigma,
                       nu = df,
                       observed = data)
      
      trace = pm.sample(1000)

      # gelman rubin
      pm.gelman_rubin(trace)

      # forestplot
      pm.forestplot(trace)

      # summary [look at mc error here. This is the std error, should be low]
      pm.summary(trace)

      #autocorrelation
      pm.autocorrplot(trace)

      # effective size
      pm.effective_n(trace)


if '__name__' == '__main__':
    main()
