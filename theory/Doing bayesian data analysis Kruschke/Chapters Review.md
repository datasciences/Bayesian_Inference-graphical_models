Overview
--------
The reviews are my learnings from Doing Bayesian Data Analysis by John K Kruschke. In each chapter I write a summay of my learnings. I also copy and pasted text from the book for my own learning purpose. Please read the original book for a better understanding.


Chapter 1
---------
The book assumes no prior knowledge of statistics but basics of probabilitty and calculus. The author covers how to read this book chapter by chapter. He lays out a plan that is good for different knowledge levels and time limits. 
The contents of the chapters are given below as given in the text book:

Chapter 2: The idea of Bayesian inference and model parameters. This chapter introduces important concepts; don’t skip it.
• Chapter 3: The R programming language. Read the sections about installing the software, including the extensive set of programs that accompany this book. The rest can be skimmed and returned to later when needed.
• Chapter 4: Basic ideas of probability. Merely skim this chapter if you have a high probability of already knowing its content.
• Chapter 5: Bayes rule!
• Chapter 6: The simplest formal application of Bayes rule, referenced throughout the remainder of the book.
4 Doing Bayesian Data Analysis
• Chapter 7: MCMC methods. This chapter explains the computing method that makes contemporary Bayesian applications possible. You don’t need to study all the mathematical details, but you should be sure to get the gist of the pictures.
• Chapter 8: The JAGS programming language for implementing MCMC.
• Chapter 16: Bayesian estimation of two groups. All of the foundational concepts from the aforementioned chapters, applied to the case of comparing two groups.

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

Assessing the properties of a target distribution by generating representative random values is a case of a Monte Carlo simulation. Any simulation that samples a lot of random values from a distribution is called a Monte Carlo simulation. The Metropolis algorithm and Gibbs sampling are specific types of Monte Carlo
process. They generate random walks such that each step in the walk is completely independent of the steps before the current position. Any such process in which each step has no memory for states before the current one is called a (first-order) Markov process, and a succession of such steps is a Markov chain (named after the mathematician Andrey Markov, 1856–1922). The Metropolis algorithm and Gibbs sampling are examples of a Markov chain Monte Carlo (MCMC) process.

Why do we use different chains with iterations rather than one? 
--------------------
Author reply on his blog: Basically, when multiple chains show the same form, we gain confidence that the chains are not stuck or clumpy in unrepresentative regions of parameter space.Page 178 onwards, second edition. 

MCMC REPRESENTATIVENESS, ACCURACY, AND EFFICIENCY
----------------------------


















