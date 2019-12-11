# Bayesian machine learning and deep learning workout area with Brms, Pymc3, Pyro and Gpytorch

Uncertainity estimation in machine learning, deep learning and reinforcement learning methods are increasingly becoming popular due to the latest research advancements in variational inference and dropout methods. 

This repository will contain book chapter reviews, bayesian model implementations and resources for learning bayesian modeling. I did a rigorous research on this topic to come up with a list of most influential books and programming packages on this topic to layout a plan for my study. The list of books and packages are given below. 

Books
--------

* Information Theory, Inference, and Learning Algorithms
* Gaussian process for machine learning
* pattern recognition and machine learning
* Statistical rethinking
* Bayesian analysis with python
* Probability theory: The logic of science
* Bayesian data analysis
* Think Bayes

If you want to learn some statistical modeling and get acquinted with statistical problem solving start with this book "https://bookdown.org/roback/bookdown-bysh/" suggested by Statistical Rethinking Record.

Reading several blogs on the web made me feel that the field(deep learning with Bayesian) is still in it's infacncy. There are several researchers and programmers working on this topic and I really hope the field will advance over time. 

Why going bayesian
--------
The main advantages of going bayesian are as follows:
* Bayesian methods typically involves using probability distributions rather than point probabilities such as mean. For example for a regression problem for predicting house prices rather than predicting the price of a house, bayesian methods produces a distribution of possible predictions. 
* Bayesian methods helps to derive credible intervals (similar to confidence interval but not the same) around using the predicted distribution around the mean. 
* Bayesian method can utilize informed or uninformed priors. Priors are nothing but prior knowledge about the distribution of samples. 
* Bayesian methods can work efficiently on small sample size, even in case of deep learning models.


Core architecture of a bayesian method
--------

Overview 
--------
There will be several folders and subfolders in this repository. 

* A summary of the most important theory will go into [[theory folder]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/theory)
* All basic statistical exercises from think bayes will go into [[stats folder]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/stats)
* All practical exercises using machine learning will go into folder [[ml]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/ml)
* All practical exercises using deep learning will go into folder [[dl]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/dl)
* Both ml and dl folders will contain subfolders with examples, inference from R and python libraries such as Pymc3, Brms, Pyro, gpytorch or Botorch. 
* ml using Pymc3
* dl using Pymc3
 

Talks
------
1) https://www.youtube.com/watch?v=HNKlytVD1Zg&t=3836s
   Bayesian logistic regression, KL divergence, neural network code : https://github.com/chrisorm/pydata-2018/tree/master/notebooks 


