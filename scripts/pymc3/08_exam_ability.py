'''
You are given an exam with 10 questions. You answered 9 correct out of 10
What is your ability in answering the questions?
'''

import pymc3 as pm

def main(n, observed):
    '''
    parameters
    --------
    n : int
        number of trials
    observed: int
         observed number of success      
    '''    
    with pm.Model() as exam_model:
      # Week uniform prior for prior
      prior = pm.Beta('prior', 0.5, 0.5)

      # Bernouli trials modeled using binomial distribution
      obs = pm.Binomial('obs', n = n, p = prior, observed = observed)

      # plot model design
      pm.model_to_graphviz(exam_model)

      # Use metropolis hasting for sampling
      step = pm.Metropolis()

      # sample from the prior distribution to get the posterior
      trace = pm.sample(5000, step)

      # plot posterior
      pm.plot_posterior(trace)

      # calculate gelman rubin stats
      pm.gelman_rubin(trace)

      
if '__name__' == '__main__':
    main()
