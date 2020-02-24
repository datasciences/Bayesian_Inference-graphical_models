'''
A repeating coin toss trial to produce a total of 5 heads through
a bernoulli distribution.

The experiment is conducted 10 times.

In each experiment, the coin need to be tossed untill a total of 5 heads are observed
Learned from : https://github.com/junpenglao/advance-bayesian-modelling-with-PyMC3/blob/master/Notebooks/Code1%20-%20Forward_Random.ipynb
'''

import scipy.stats as st
import numpy as np
import matplotlib.pylab as plt

def main(n_exp = 10, p = 0.5, init_size = 5):
  
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

if '__name__' == '__main__':
    main()
