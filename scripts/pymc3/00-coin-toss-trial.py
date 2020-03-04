import scipy.stats as st
import numpy as np
import matplotlib.pylab as plt

def main(n_exp = 10, p = 0.5, init_size = 5):
    '''
    A function to generate several coin toss experiments. 

    The experiment needed to be conducted n time.
    
    In each experiment, the coin is tossed untill a total
    of 5 heads are observed

    Learned from : https://github.com/junpenglao/advance-bayesian-modelling-with-PyMC3/blob/master/Notebooks/Code1%20-%20Forward_Random.ipynb

    parameters
    ----------
    n_exp : int
            Number of experiments
    p     : float
            Prior probability of coin
    init_size : int
            Number of heads to be observed

    '''
  
    # conduct the experiment 10 times
    for i in range(n_exp):

        # draw from a bernoulli trial with p = 0.5 and size 5
        outcome = st.bernoulli.rvs(p = p, size = init_size)    

        # continue the experiment untill you head 5 heads
        while outcome.sum() != init_size:
            outcome = np.append(outcome, st.bernoulli.rvs(p = p))

        # print the length and outcome of each experiment
        # based on the length of the outcome 
        if len(outcome)<10:
            print('', len(outcome), '<---', outcome)

        else:
            print(len(outcome), '<---', outcome)

if '__name__' == '__main__':
    main(n_exp = 10, p = 0.5, init_size = 5)
