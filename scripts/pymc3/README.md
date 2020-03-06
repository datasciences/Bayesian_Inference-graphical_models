# Pymc3 random concepts
Ref: https://docs.pymc.io/developer_guide.html


To understand the concepts in pymc3 and ppl in general, it is important to understand the logic behind it. Read the following links to get an overview of what goes under the hood of a ppl.
1. http://alfan-farizki.blogspot.com/2015/07/pymc-tutorial-bayesian-parameter.html
2. https://github.com/ericmjl/bayesian-analysis-recipes/blob/master/notebooks/sampling-loglikelihood.md


What is logp?
-----
Calling myModel.logp returns the log probability of the model at the current parameter setting. So if you ran MAP() it would be the logp at the MAP, if you sampled, it would be the logp at the last sampled parameter value.

Ex: pm.Normal is a density function of a standard normal
pm.Normal.dist(mu = 0, sigma = 1, shape = 2).logp(0).eval()

What is a random variable distribution definition do?
------
distributions in pymc3 has two methods, random and logp. random generates data points using numpy and scipy whereas logp generates log probability value using theano tensors. 

logp method
-------
logp calculates the logposterior of the free variables and data. A notebook is included to further study this concept (08_developer-guide-03).

Write a standalone distribution
-----
y = pm.Binomial.dist(n=10, p=0.5)
y.logp(4).eval()

Dirichlet distribution
------
Dirichlet is a generalization of beta distribution with parameter alpha (Beta distribution has parameter alpha and beta since it has only two outcomes). Dirichlet will have several alpha parameters representing each variable and their corresponding density. It is a conjugate prior of multinomial distribution, similar to beta distribution being the conjugate prior of the binomial distribution. 


A wonderful explanation is given here:https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution 


How does a Normal randomvariable function looks like?
-------

class Normal(Continuous):

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0) (only required if tau is not specified).
    tau : float
        Precision (tau > 0) (only required if sigma is not specified).

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.Normal('x', mu=0, sigma=10)

        with pm.Model():
            x = pm.Normal('x', mu=0, tau=1/23)
    """

    def __init__(self, mu=0, sigma=None, tau=None, sd=None, **kwargs):
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)
        self.tau = tt.as_tensor_variable(tau)

        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.variance = 1. / self.tau

        assert_negative_support(sigma, 'sigma', 'Normal')
        assert_negative_support(tau, 'tau', 'Normal')

        super().__init__(**kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Normal distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, tau, _ = draw_values([self.mu, self.tau, self.sigma],
                                 point=point, size=size)
        return generate_samples(stats.norm.rvs, loc=mu, scale=tau**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Normal distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        sigma = self.sigma
        tau = self.tau
        mu = self.mu

        return bound((-tau * (value - mu)**2 + tt.log(tau / np.pi / 2.)) / 2.,
                     sigma > 0)



log posterior
-------
Ref : https://github.com/pymc-devs/pymc3/blob/7493d5b61eeff58120f0d0e8b6cfbc05556c565b/pymc3/stats.py#L119


def _log_post_trace(trace, model=None, progressbar=False):
    """Calculate the elementwise log-posterior for the sampled trace.
    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion
    Returns
    -------
    logp : array of shape (n_samples, n_observations)
        The contribution of the observations to the logp of the whole model.
    """
    model = modelcontext(model)
    cached = [(var, var.logp_elemwise) for var in model.observed_RVs]

    def logp_vals_point(pt):
        if len(model.observed_RVs) == 0:
            return floatX(np.array([], dtype='d'))

        logp_vals = []
        for var, logp in cached:
            logp = logp(pt)
            if var.missing_values:
                logp = logp[~var.observations.mask]
            logp_vals.append(logp.ravel())

        return np.concatenate(logp_vals)

    try:
        points = trace.points()
    except AttributeError:
        points = trace

    points = tqdm(points) if progressbar else points

    try:
        logp = (logp_vals_point(pt) for pt in points)
        return np.stack(logp)
    finally:
        if progressbar:
            points.close()

A glimpse into categorical distribution
-----
Categorical distribution is used for individual data points distribution modeling. It accepts Dirichlet distribution proportions and total number of data points in the observation. 

It is given by:
zi|pk ∼Cat(p) 

In pymc3 it is used to model Zi | Pk, where Zi are categories of each data point given the probability of the data point Pk being from any of the k mixtures.  

zi are individual group assignments for data points with probability of being from either one of K components. Where K components are modeled using dirichlet distributions. The dirichlet distribution models multivariate data where the proportion sums to 1. 

A free categorical random variable in pymc3 always gives outputs as 0s. This is bit strange at this point for me.

Mixture model
------
A mixture model accepts weight components w and component distributions. 
w can be modeled using dirichlet distribution.
component distribution can be modeled using Multivariate distribution with mean mu and sigma. Mixture model shape usually represents the number of components in the mixture. 

Multivariate distribution
-----
Multivariate model accepts a vector of size n for the 'mean' and a covariance matrix. It is also possible to give a iterable mean and iterable covariance matrices to model independent group of variables. The shape of the Multivariate Normal can be equal to the number of features in the multivariate distribution. 

Dirichlet distribution
-------
It is used to model proportions. It has only one parameter alpha where alpha > 1 and the sum of proportion would be 1.  


basic, free and observed random variables
-----
Once a model is defined, one can always call model.Free_RVs or model.Observed_Rvs to get the random variables in the model.

Shape parameter
-----
It is important to understand that none of the random variables have any shape parameter. It is a parameter used by pymc3 to evaluate a function a

Observed random variable
------
Observed argument explicitly ﬂags the random variable 'Obs'(User defined) as the one that is not a latent variable that can vary across model simulations (i.e., it will not be estimated), but instead is given by the data. 