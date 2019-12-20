# Random basic notes


Uncertainity estimation in machine learning, deep learning and reinforcement learning methods are increasingly becoming popular due to the latest research advancements in variational inference and dropout. Reading several blogs on the web made me feel that the field(deep learning with Bayesian) is still in it's infacncy, although a lot of talented researchers, statisticians and programmers are working hard to cater the advantages of bayesian estimation to a wider audiene. 

With the availability of several Python and R packages and bayesian statistics books, it is good to learn the theory, then practice and implement solutions for already existing problems. I envision to use the theory and practical knowledge I would gain on projects my company would work in the future. 

After an extensive search, I found the following books are highly popular in this field. 

* Information Theory, Inference, and Learning Algorithms
* Gaussian process for machine learning
* pattern recognition and machine learning
* Statistical rethinking
* Bayesian analysis with python
* [[Probability theory: The logic of science]](https://github.com/terryum/awesome-deep-learning-papers)
* Bayesian data analysis
* Think Bayes




Disclaimer:  The content and words below are my learnings from the book "Think Bayes". Many of the contents below would be a "copy paste from Think bayes book" or other websites for my own understanding. Please read the original book to get an indepth understanding. 


## Day 1

I have been working on building and analysing bayesian models over a month by today. I am currently using r package brms to build bayesian regression model and interpret the results. As I tried to dig the chapters deeper, i came to realize that there are several topics in bayesian statistics that I have no clue about. My real aim of delving in to bayesian field was that I wanted to learn and use bayesian deep learning models in my company projects. I spent several hours going through different blogs, text books and youtube videos. I skimmed through brms, Pymc3, Pyro and gpytorch official website tutorials. 

I'm currently working on reading Think Bayes book and complementing the theoretical part randomly using "pattern recognition and machine learning", "Information Theory, Inference, and Learning Algorithms" books.

On day 1, I completed most of the theory and practical part of chapter 1 of Think Bayes. 

Some of the important topics I covered are 
1) Conditional probability : Cases where the events are not mutually exclusive. 

Conditional probability is the probability of one event occurring with some relationship to one or more other events. For example:

Event A is that it is raining outside, and it has a 0.3 (30%) chance of raining today.
Event B is that you will need to go outside, and that has a probability of 0.5 (50%).
A conditional probability would look at these two events in relationship with one another, such as the probability that it is both raining and you will need to go outside.

Ref : https://www.statisticshowto.datasciencecentral.com/what-is-conditional-probability/

2) Conjoint probability: The probability that two things are true.

Suppose that A means that it rains today and B means that it rains tomorrow. If I know that it rained today, it is more likely that it will rain tomorrow, so p(B|A) > p(B).

In general, the probability of a conjunction is

p(A  and  B) = p(A) p(B|A) 

Ref: http://www.greenteapress.com/thinkbayes/html/thinkbayes002.html

3) Bayes theorem

4) The cookie problem.

5) The diachronic interpretation : Update the probability of a hypothesis, H, in light of some body of data, D.

This way of thinking about Bayes’s theorem is called the diachronic interpretation. “Diachronic” means that something is happening over time; in this case the probability of the hypotheses changes, over time, as we see new data.

Rewriting Bayes’s theorem with H and D yields:

p(H|D) = 	
p(H) p(D|H)
p(D)
 
In this interpretation, each term has a name:

p(H) is the probability of the hypothesis before we see the data, called the prior probability, or just prior.
p(H|D) is what we want to compute, the probability of the hypothesis after we see the data, called the posterior.
p(D|H) is the probability of the data under the hypothesis, called the likelihood.
p(D) is the probability of the data under any hypothesis, called the normalizing constant. Normalizing constant is a weighting factor, adjusting the odds towards the more likely outcome.Read this example for a more thorough understanding. 


6) The M and M problem
 	Prior	Likelihood	 	Posterior
 	p(H)	p(D|H)	p(H) p(D|H)	p(H|D)
A	1/2	(20)(20)	200	20/27
B	1/2	(14)(10)	70	7/27


The problem is very interesting. It was solved in the book using Bayes theorem 

p(H|D) = p(H) p(D|H) / p(D)

However, it is not yet clear to me why P(D) that is the normalizing constant (Probability of seeing the data under any hypothesis) is a whole number 270? Instead of a probability ie may be P(Y|Bag1 and Bag2).

Update: After going through the beginning of chapter 3, I understood the concept of normalizing. 
Probability of seeing the data under any hypothesis is nothing but the probability of an event happening without having any hypothesis.

ex: probability of head or tail during a coin tos is 0.5
    probability of getting a toss from a 4, 6, 8, 12, and 20 sided die is 1/5 = 0.20
    
    With the example from chapter 3 dice throw
    
    class Dice(Pmf):    
    def __init__(self, hypos):
        """Initialize self.
        """
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, 1)
        self.Normalize() # Normalizing probability

    def Update(self, data):
        """Updates the PMF with new data.
        
        """
        for hypo in self.Values():
            self[hypo] *= self.Likelihood(data, hypo)
        self.Normalize()
        
        for hypo, prob in self.Items():
            print(hypo, prob)
            
    def Likelihood(self, data, hypo):
        """The likelihood of the data under the hypothesis.        
        """
        if hypo < data:
            return 0
        else:
            return 1.0/hypo
    
pmf = Dice([4, 6, 8, 12, 20])
pmf.Update(6)
#pmf.Print()






The exercises from this section taken from Think Bayes github repository can be accessed from [[here]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/blob/master/stats/day2.ipynb)
 

[[Original git hub repo]](https://github.com/rlabbe/ThinkBayes)

## Day 2

Today I started reading about Variational inference and how it is faster than Markov chain monte carlo approach for sampling. 

[[Variational inference a primer]] (https://fabiandablander.com/r/Variational-Inference.html#fnref:1)
[[blog]](https://www.r-bloggers.com/a-brief-primer-on-variational-inference/)

I started by reading this paper on [[Bayesian Uncertainty Quantification with Synthetic Data]](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/sites/ca.waterloo-intelligent-systems-engineering-lab/files/uploads/files/waise.pdf)

## Abstract
Image semantic segmentation systems based on deep learning are prone to making erroneous predictions for images affected by uncertainty influence factors such as occlusions or inclement weather. Bayesian deep learning applies the Bayesian framework to deep models and allows estimating so-called epistemic and aleatoric uncertainties as part of the prediction. Such estimates can indicate the likelihood of prediction errors due to the influence factors. However, because of lack of data, the effectiveness of Bayesian uncertainty estimation when segmenting images with varying levels of influence factors has not yet been systematically studied. In this paper, we propose using a synthetic dataset to address this gap. We conduct two sets of experiments to investigate the influence of distance, occlusion, clouds, rain, and puddles on the estimated uncertainty in the segmentation of road scenes. The experiments confirm the expected correlation between the influence factors, the estimated uncertainty, and accuracy. Contrary to expectation, we also find that the estimated aleatoric uncertainty from Bayesian deep models can be reduced with more training data. We hope that these findings will help improve methods for assuring machine-learning-based systems.

## How to make a an existing neural network bayesian?

Deep models contain a large number of weights, applying the Bayesian framework to a deep model is computationally intractable, therefore, in order to obtain different sets of weight values, we need to use Bayesian approximation techniques.

One approach to obtain an approximated BNN model from an existing DNN architecture is by inserting dropout layers and training the new model with
dropout training [8]. At test time, for a given input, we perform multiple forward predictions in the network while keeping the dropout layers active. In other words, we remove a percentage of randomly-selected units (i.e., set the weight values of their connections to 0) from the trained model in order to obtain a sample prediction for the given input; then we repeat this process T times and calculate the average prediction. This technique at test time is referred to as Monte-Carlo (MC) dropout.


Uncertainity in deep learning
----------------
[[Two types of uncertainty quantification problems]](https://en.wikipedia.org/wiki/Uncertainty_quantification#Aleatoric_and_epistemic_uncertainty)

Epistemic uncertainty, also known as model uncertainty represents what the model does not know due to insufficient training data. This kind of uncertainty can be explained away with more training data. Aleatoric uncertainty is caused due to noisy measurements in the data and can be explained away with increased sensor precision (but cannot be explained away with increase in training data)

There are two major types of problems in uncertainty quantification: one is the forward propagation of uncertainty (where the various sources of uncertainty are propagated through the model to predict the overall uncertainty in the system response) and the other is the inverse assessment of model uncertainty and parameter uncertainty (where the model parameters are calibrated simultaneously using test data). There has been a proliferation of research on the former problem and a majority of uncertainty analysis techniques were developed for it. On the other hand, the latter problem is drawing increasing attention in the engineering design community, since uncertainty quantification of a model and the subsequent predictions of the true system response(s) are of great interest in designing robust systems.

Forward uncertainty propagation
See also: Propagation of uncertainty
Uncertainty propagation is the quantification of uncertainties in system output(s) propagated from uncertain inputs. It focuses on the influence on the outputs from the parametric variability listed in the sources of uncertainty. The targets of uncertainty propagation analysis can be:

To evaluate low-order moments of the outputs, i.e. mean and variance.
To evaluate the reliability of the outputs. This is especially useful in reliability engineering where outputs of a system are usually closely related to the performance of the system.
To assess the complete probability distribution of the outputs. This is useful in the scenario of utility optimization where the complete distribution is used to calculate the utility.
Inverse uncertainty quantification
See also: Inverse problem
Given some experimental measurements of a system and some computer simulation results from its mathematical model, inverse uncertainty quantification estimates the discrepancy between the experiment and the mathematical model (which is called bias correction), and estimates the values of unknown parameters in the model if there are any (which is called parameter calibration or simply calibration). Generally this is a much more difficult problem than forward uncertainty propagation; however it is of great importance since it is typically implemented in a model updating process. There are several scenarios in inverse uncertainty quantification.


# Weight Uncertainty in Neural Networks [[paper]](https://arxiv.org/pdf/1505.05424.pdf)

We introduce a new, efficient, principled and backpropagation-compatible algorithm for learning a probability distribution on the weights of a neural network, called Bayes by Backprop. It regularises the weights by minimising a compression cost, known as the variational free energy or the expected lower bound on the marginal likelihood. We show that this principled kind of regularisation yields comparable performance to dropout on MNIST classification. We then demonstrate how the learnt uncertainty in the weights can be used to improve generalisation in non-linear regression problems, and how this weight uncertainty can be used to drive the exploration-exploitation trade-off in reinforcement learning.

# Knowing What You Know in Brain Segmentation Using Bayesian Deep Neural Networks
[[paper]](https://www.frontiersin.org/articles/10.3389/fninf.2019.00067/full)

Data and Tensorflow code is available, Used Meshnet

Abstract

One of the most popular approximate inference methods for neural networks is variational inference, since it scales well to large DNNs. In variational inference, the posterior distribution [Math Processing Error] is approximated by a learned variational distribution of weights qθ(w), with learnable parameters θ. This approximation is enforced by minimizing the Kullback-Leibler divergence (KL) between qθ(w), and the true posterior, [Math Processing Error], [Math Processing Error], which measures how qθ(w) differs from [Math Processing Error] using relative entropy. This is equivalent to maximizing the variational lower bound (Hinton and Van Camp, 1993; Graves, 2011; Blundell et al., 2015; Kingma et al., 2015; Gal and Ghahramani, 2016; Louizos and Welling, 2017; Molchanov et al., 2017), also known as the evidence lower bound (ELBO).


Gaussian process
------------
cess.
The Gaussian process (GP) is a powerful tool in statistics that allows us to model distributions over functions (Rasmussen and Williams)


ARA: accurate, reliable and active histopathological image classification framework with Bayesian deep learning
-----------------------------------------
[[paper]](https://www.nature.com/articles/s41598-019-50587-1)

Abstract
Machine learning algorithms hold the promise to effectively automate the analysis of histopathological images that are routinely generated in clinical practice. Any machine learning method used in the clinical diagnostic process has to be extremely accurate and, ideally, provide a measure of uncertainty for its predictions. Such accurate and reliable classifiers need enough labelled data for training, which requires time-consuming and costly manual annotation by pathologists. Thus, it is critical to minimise the amount of data needed to reach the desired accuracy by maximising the efficiency of training. We propose an accurate, reliable and active (ARA) image classification framework and introduce a new Bayesian Convolutional Neural Network (ARA-CNN) for classifying histopathological images of colorectal cancer. The model achieves exceptional classification accuracy, outperforming other models trained on the same dataset. The network outputs an uncertainty measurement for each tested image. We show that uncertainty measures can be used to detect mislabelled training samples and can be employed in an efficient active learning workflow. Using a variational dropout-based entropy measure of uncertainty in the workflow speeds up the learning process by roughly 45%. Finally, we utilise our model to segment whole-slide images of colorectal tissue and compute segmentation-based spatial statistics.


Variational dropout for inference and uncertainty estimation
In order to provide more accurate classification as well as uncertainty prediction, we adopted a popular method called variational dropout51. The central idea of this technique is to keep dropout enabled by performing multiple model calls during prediction. Thanks to the fact that different units are dropped across different model calls, it might be considered as Bayesian sampling from a variational distribution of models24. In a Bayesian setting, the parameters (i.e. weights) ω of a CNN model are treated as random variables. In variational inference, we approximate the posterior distribution P(ω|D) by a simpler (variational) distribution q(ω), where D is the training dataset. Thus, we assume that ω^t∼q(ω), where ω^t is an estimation of ω resulting from a variatonal dropout call t. With these assumptions, the following approximations can be derived. 

Variational drop out paper is given below

BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
[[paper]](https://arxiv.org/pdf/1506.02158.pdf)

Thesis that summarizes all the topics in bayesiandeep learning, variational inference
------------------------
https://omegafragger.github.io/files/ociamthesismain.pdf


Uncertainity estimation ml and dl papers and code
-------------------
https://github.com/ahmedmalaa/uncertainty


Day 3
--------------
Incredible resources for deep learning with code
1) https://forums.fast.ai/t/uncertainty-in-deep-learning-bayesian-networks-gaussian-processes/5551/6
2) https://towardsdatascience.com/bayesian-deep-learning-with-fastai-how-not-to-be-uncertain-about-your-uncertainty-6a99d1aa686e
3) https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/https://colab.research.google.com/drive/1bIFAWl_o__ZKFSVvE9994WJ2cfD9976i
4) https://colab.research.google.com/drive/1bIFAWl_o__ZKFSVvE9994WJ2cfD9976i
5) https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/

Most important papers in this filed

[1] Improving neural networks by preventing co-adaptation of feature detectors, G. E. Hinton, et al., 2012
[2] Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, Y. Gal, and Z. Ghahramani, 2016
[3] Dropout: A Simple Way to Prevent Neural Networks from Overfitting, N. Srivastava, et al., 2014


Day 4
------------
Overview
--------
The reviews are my learnings from Doing Bayesian Data Analysis by John K Kruschke. In each chapter I write a summay of my learnings. I also copy and pasted text from the book for my own learning purpose. Please read the original book for a better understanding.


Chapter 1
---------
The book assumes no prior knowledge of statistics but basics of probabilitty and calculus. The author covers how to read this book chapter by chapter. He lays out a plan that is good for different knowledge levels and time limits. 
The contents of the chapters are given below as given in the text book:

1. Chapter 2: The idea of Bayesian inference and model parameters. This chapter introduces important concepts; don’t skip it.
2. Chapter 3: The R programming language. Read the sections about installing the software, including the extensive set of programs that accompany this book. The rest can be skimmed and returned to later when needed.
3. Chapter 4: Basic ideas of probability. Merely skim this chapter if you have a high probability of already knowing its content.
4. Chapter 5: Bayes rule!
5. Chapter 6: The simplest formal application of Bayes rule, referenced throughout the remainder of the book.
6. Chapter 7: MCMC methods. This chapter explains the computing method that makes contemporary Bayesian applications possible. You don’t need to study all the mathematical details, but you should be sure to get the gist of the pictures.
7. Chapter 8: The JAGS programming language for implementing MCMC.
8. Chapter 16: Bayesian estimation of two groups. All of the foundational concepts from the aforementioned chapters, applied to the case of comparing two groups.

Table 1 in chapter 1 also illustrates the Bayesian analogues of null hypothesis significance tests such as binomial tests, t-test etc.

There are links to R and Python implementation and workshops, however they do not work anymore, so don't get disappointed. You should be able to find the links on github, if you do a rigorous search. It is not relevant at this moment for me. I will add it here later, if I get a chance to implement the practicals. But for now I'm trying to understand the core concepts in detail.

Chapter 2
---------
Introduction: Credibility, Models, and Parameters

Chapter 2 goes through the basics of bayesian such as prior and posterior probabilites. You can skip this section if you already know these topics. I personally found the examples he used very simplistic and understandable, so I recommend reading it. 

The first take away from the first section of the chapter is "Bayesian Inference Is Reallocation of Credibility Across Possibilities". This means when you start solving a problem using bayesian methods, you start with a hypothesis. As you get more and more data or evidence you update the prior beliefs and you get the posterior beliefs. Since we are dealing with data, you say "Prior distribution" and "Posterior distribution". One of the advantage of using a distribution rather than a mean in bayesian methods is that, you account for variability in the measurement of the data. For example: You are measuring the blood pressure of several patients, the blood pressure of the same patient can vary minutely depending on several factors. So having a model that can account for this variability is always great.

A bayesian data analysis follows these steps:
1. Identifying the data relevant to the questions. 
2. Define a descriptive model for the data. 
3. Specify a prior distribution.
4. Use Bayesian inference to re-allocate credibility across parameter values and interpret the posterior distribution.
5. Check that the posterior predictions mimic the data with reasonable accuracy.

Problem 1: Modeling human weight as a function of height

We assume predicted weight is a multiplier times height plus a baseline. 
yˆ = β1x + β0

The model is not complete yet, because we have to describe the random variation of actual weights around the predicted weight.For simplicity, we will use the conventional normal distribution (explained in detail in Section 4.3.2.2), and assume that actual weights y are distributed randomly according to a normal distribution around the predicted value yˆ and with standard deviation denoted σ (Greek letter “sigma”). This relation is denoted symbolically as
y ∼ normal(yˆ, σ )

where the symbol “∼” means “is distributed as.” The above equation says that values near yˆ are most probable, and y values higher or lower than yˆ are less probable. The decrease in probability around yˆ is governed by the shape of the normal distribution with width specified by the standard deviation σ.
The full model, combining above equatios has three parameters altogether: the slope, β1, the intercept, β0, and the standard deviation of the “noise,” σ. Note that the three parameters are meaningful. In particular, the slope parameter tells us how much the weight tends to increase when height increases by an inch, and the standard deviation parameter tells us how much variability in weight there is around the predicted value. The third step in the analysis is specifying a prior distribution on the parameters.

The third step in the analysis is specifying a prior distribution on the parameters. The author used a noncommittal and vague prior that places
virtually equal prior credibility across a vast range of possible values for the slope and intercept, both centered at zero. He also placed a vague and noncommittal prior on the noise (standard deviation) parameter, specifically a uniform distribution that extends from zero to a huge value. T

The fourth step is interpreting the posterior distribution. Bayesian inference has reallocated credibility across parameter values, from the vague prior distribution, to values that are consistent with the data. The posterior distribution indicates combinations of β0, β1, and σ that together are credible, given the data. 



Chapter 7
---------
Markov chain monte carlo (MCMC)

Assessing the properties of a target distribution by generating representative random values is a case of a Monte Carlo simulation. Any simulation that samples a lot of random values from a distribution is called a Monte Carlo simulation. The Metropolis algorithm and Gibbs sampling are specific types of Monte Carlo process. They generate random walks such that each step in the walk is completely independent of the steps before the current position. Any such process in which each step has no memory for states before the current one is called a (first-order) Markov process, and a succession of such steps is a Markov chain (named after the mathematician Andrey Markov, 1856–1922). The Metropolis algorithm and Gibbs sampling are examples of a Markov chain Monte Carlo (MCMC) process.

*Markov chain monte carlo

![GitHub Logo](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/blob/master/images/MCMC.PNG)
Markov chain monte carlo consists of three parts. 
1. Monte carlo
2. Markov chain
3. Decision making

In the first step, using Monte carlo method random variable can be drawn from an assigned prior distribution. 
In the second step, we make use of a markov chain. Markov chain is nothing but a stochastic process in which a chain of random variables are drawn based on the target distribution and the optimizing function. The chain is run for 2000 or more interations to search the parameter distribution space for random sampling. You have to run the chain several times (defaults to 4 chains in most packages) and compare the behaviour of the chains to understand if the convergence of the algorithm is good enough. For an initial time period called "warm-up phase" the chain tries to find optimal path and stabilizes. Once it stabilizes, the chain continues and you call it sampling phase. Usually the warm up phase is discarded. 
In the third step of decision making, each time when the markov chain randomly sample a new point from the parameter space, the ratio betweeen the specified (likelyhood * prior) for both t and t-1 is calculated. Further, a uniform random variable is drawn to check if the ratio is greater than the uniform random variable. If the ratio is higher the new random sample drawn by the markov chain is accepted. The process runs untill the posterior distribution approximates the target distribution. 

Why do we use different chains with iterations rather than one? 
--------------------
Author reply on his blog: Basically, when multiple chains show the same form, we gain confidence that the chains are not stuck or clumpy in unrepresentative regions of parameter space.Page 178 onwards, second edition. 

MCMC REPRESENTATIVENESS, ACCURACY, AND EFFICIENCY

In principle, the mathematics of MCMC guarantee that infinitely long chains will achieve a perfect representation of the posterior distribution, but unfortunately we do not have infinite time or computer memory.


MCMC diagnostics

The first method to detect unrepresentativeness is a visual examination of the chain trajectory. A graph of the sampled parameter values as a function of step in the chain is called a trace plot.

The preliminary steps, during which the chain moves from its unrepresentative initial value to the modal region of the posterior, is called the burn-in period. For realistic applications, it is routine to apply a burn-in period of several hundred to several thousand steps. 


 # Day 5
Completed watching Markov chain explanation, MCMC diagnostics, MCMC posterior checks and reading page numbers 180-184 from doing bayesian data analysis. 
Went through several discussion forums to understand the core difference and advantages of each programming language such as Pymc3, Pyro and Gpytorch. 
It has come to my understanding that Pymc3 uses theano underthe hood. They have plans to change the engine with either tensorflow, pytorch or pyro in the newer version Pymc4 I guess. One of the difficulty they say in switching to pytorch is it uses dynamic computational graphs wheras as bayesian models of pymc3 is better suited for static computational graphs. 

Pyro and gpytorch uses pytorch as it's engine and both has its advantages. Gpytorch is a deep bayesian kernal method whereas Pyro is a deep bayesian probabilistic frame work. There are some examples in the documentation of gpytorch that made use of SVI, ELBO and Adam from pyro. It is not clear why simple variational inference are executed using Pyro in gpytorch. It could be that it is an example to show how both frameworks can be combined. 

I decided to spend my time reading statistical rethiking and follow the code in the repository https://fehiepsi.github.io/rethinking-pyro/08-markov-chain-monte-carlo.html. This book has the advantage that it has implementations available chapterwise for brms, pyro and pymc3. It would be easy to switch between frameworks if I don't understand something. 

From the discussion below it seems that Pyro has some advantage. Ref: https://github.com/pyro-ppl/pyro/issues/1265

fehiepsi: It was a great conversation! Because I didn't say much during the talk, I'd like to take this chance to share some of Pyro GP's features which might be helpful for your plan:
Beside SVI, Pyro GP also supports HMC.
For deep kernel learning, Cornell GPytorch uses low dimensional outputs of a pre-trained neural network to feed into a GP model. On the other hand, Pyro GP can train both network's parameters and GP's hyperparameters at the same time. For example, https://github.com/uber/pyro/blob/dev/examples/contrib/gp/sv-dkl.py illustrates how to achieve it. It just takes a few lines of code to make that combination (lines 97, 98, and 103). The speed is also fast because things are trained in mini-batch.
The mean function is also flexible to define. It can be a neural network, and you can use SVI to learn its parameters. This is useful when we want to define a deep GP model and want to make this mean function play the role of 'skip layer' as in the doubly stochastic SVI paper. After my vacation, I'll write a tutorial replicating that paper to illustrate how to compose GP models using Pyro GP.
We can seamlessly set priors/constraints for hyperparameters using PyTorch/Pyro's distributions and distributions.constraints modules.
If we want to fix a hyperparameter (e.g. lengthscale), we just call kernel.fix_param("lengthscale").
Traditional sparse GPR models such as FITC, DTC are available. But I put more weights on variational sparse approach because it can be trained in mini-batch and is compatible to arbitrary likelihood.
From our discussion, I guess what currently lacked from Pyro GP are:
KISSGP + LOVE, which has been well developed in Cornell GPytorch.
Using CG solve (in addition to the current Cholesky + triangle solve).
More sophisticated kernel configurations from Uber GPytorch.
I have read the GPytorch's GP regression tutorial. It seems the first thing to do is to use pyro.module/pyro.random_model on model to register its parameter or Pyro.optim. The loss can be added to a Pyro model to be learned under SVI/HMC using Bernoulli trick. Then, the next thing is to replace/inherit gpytorch.priors.SmoothedBoxPrior to make it compatible with PyTorch/Pyro's Distribution (so log_lengthscale_prior can be learned using pyro.param/pyro.sample).


@jacobrgardner: Just to clarify, GPyTorch absolutely trains the network parameters and GP hyperparameters simultaneously. This is largely the point and strength of DKL. In the MNIST example we start by pretraining just to show it is possible, but then the model is trained jointly. In the LOVE + SKI notebook we demonstrate training a deep kernel model from scratch, and we have several DKL CIFAR10 and CIFAR100 models trained end to end from scratch.
Edit: If it is helpful, I just added a tutorial on training a DKL + DenseNet model from scratch on CIFAR10 and CIFAR 100 at https://github.com/cornellius-gp/gpytorch/blob/master/examples/DKL_DenseNet_CIFAR_Tutorial.ipynb 😃


Sampling Hyperparamters with GPyTorch + NUTS
-----------------
1. https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Simple_GP_Regression/Simple_GP_Regression_Fully_Bayesian.ipynb
2. http://pyro.ai/numpyro/gp.html


All gpytorch examples can be found here. 
https://github.com/cornellius-gp/gpytorch/tree/master/examples


Day 6
------------
On the 6th day, I feel very confused. Yes, Going through several materials at this pace can be overwhelming. Though I wanted to start learning Pyro and gpytorch, pymc3 commands are more readable and understandable. At this moment I am feeling to stick with pymc3 just because i wanted to understand the concepts in detail rather than a package, Be it sampling, the challenges in drawing sampling,  SVI,  NUTS or MCMC, Pymc3 has good tutorials with explanations for my level of understanding. So let's see some pymc3 examples:

https://www.marsja.se/probabilistic-programming-in-python/


Day 7
-------
A univariate distribution refers to the distribution of a single random variable. On the other hand, a multivariate distribution refers to the probability distribution of a group of random variables. For example, a multivariate normal distribution is used to specify the probabilities of returns of a group of n stocks. This has relevance because the returns of different stocks in the group influence each other’s behaviour, that is, the behaviour of one random variable in the group is influenced by the behaviour of another variable.

How to Construct Multivariate Distribution?

For discrete random variables, joint probabilities are used to describe the multivariate distribution

For continuous random variables, if each random variable follows a normal distribution, a multivariate normal distribution is created. Remember that a linear combination of 2 or more normally distributed random variables is also normally distributed.

If we want to describe the multivariate normal distribution of the returns of a group of stocks, we need the following three parameters:

List of means returns of each stock
List of variances of returns of each stock
List of correlations between each pair of stocks.

A univariate normal distribution is described using just the two variables namely mean and variance. For a multivariate distribution we need a third variable, i.e., the correlation between each pair of random variables. This is what distinguishes a multivariate distribution from a univariate distribution. If there are n random variables in the group, we will have n*(n-1)/2 pairs of correlations.

Distributions in statistics [https://financetrain.com/series/distributions-frm/] [Full Tutorials: https://financetrain.com/series/]

Law of large numbers. 


Day 8
---------
I was really busy with revising all the topics I have studies so far and gettting a better intution behind all the algorithms that I learned. I decided to focus on text books "Doing Bayesian data analysis" and "Statistical rethinking". Both the books have examples in Pymc3. I build a couple of models in pymc3 making use of the knowledge I learned from different resources. I also read chapter 29 of Information theory, inference and learning algorithms. In the next 2 weeks, I plan to get a thorough understanding of probability theory mathematics part, try out few machine learning problems using pymc3 by reading the books I mentioned. 


Enricho suggested to listen to this video: https://www.youtube.com/watch?v=R6d-AbkhBQ8

