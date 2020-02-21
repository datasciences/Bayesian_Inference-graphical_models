'''
A repeating coin toss trial to produce a total of 5 heads.
The trial is conducted for 10 consecutive times.
'''

import scipy.stats as st
import numpy as np
import matplotlib.pylab as plt

def coin_toss_experiment(n_exp = 10, p = 0.5, init_size = 5):
  
    n_exp = 10

    # conduct the experiment 10 times
    for i in range(n_exp):

        # draw from a bernoulli trial with p = 0.5 and size 5
        outcome = st.bernoulli.rvs(p = 0.5, size = 5)    

        # continue the experiment untill you head 5 heads
        while outcome.sum() != 5:
            outcome = np.append(outcome, st.bernoulli.rvs(p = 0.5))

        # print the length and outcome of each experiment
        # based on the length of the outcome 
        if len(outcome)<10:
            print('', len(outcome), '<---', outcome)

        else:
            print(len(outcome), '<---', outcome)
