import scipy.stats as st
import numpy as np
import matplotlib.pylab as plt

def main(n_exp = 10, p = 0.9, init_size = 5):
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
    # Variable to store outputs of experiments
    X = []

    for i in range(n_exp):
        outcome = st.bernoulli.rvs(p = p, size = init_size)

        while outcome.sum()!= init_size:
            outcome = np.append(outcome, st.bernoulli.rvs(p = p))
        X.append(len(outcome))


    plt.hist(np.asarray(X),
             np.arange(min(X), max(X)),
             rwidth = 0.75,
             density = True)


if '__name__' == '__main__':
    main(n_exp = 10, p = 0.5, init_size = 5)
