# Bayesian machine learning and deep learning workout area with Brms, Pymc3, Pyro and prophet.

![GitHub Logo](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/blob/master/images/Bayesian.jpg)


This serves as a repository containing my Bayesian Inference learnings. Please do not get overwhelmed by the names of several packages written in the title. During my learning curve, due to the resources I used, I moved from brms through pymc3 to pyro. Initially, when I started out, I built models in brms (An R package). It was quite to easy to build and improve the model. But, eventually when it got harder to assess the outputs, underlying theory and a need for bayesian python models, I switched to Pymc3, which has a strong open source community around it. Later, once I picked up the theory and started building models in pymc3, I switched myself to pyro and trying my best to get expertise in it. You can read more about bayesian inference and the organisation of my resources below. 

Why I made this repository?

Uncertainity estimation in machine learning, deep learning and reinforcement learning methods are increasingly becoming popular due to the latest research advancements in variational inference, monte carlo dropout, bayes by backpropagation and its application in convolutional and recurrent neural networks. 

This repository will contain book chapter reviews, bayesian model implementations and resources for learning bayesian modeling. I did a rigorous research on this topic to come up with a list of most influential books and programming packages on this topic to layout a plan for my study. The list of books and packages are listed under Books. 

Overview 
--------
There will be several folders and subfolders in this repository. Currently, this is a compilation of several knowledge that I learned from different resoures. If you would like to follow along, I would suggest you to start with Think Bayes book and then stick to Statistical Rethinking book. Further, you can access exercises from "notebooks folder". For book chapters review, I have really enjoyed reading from different authors since it helped me to understand and reason about the same concept multiple time. Currently, I am sticking to statistical rethinking book, but I do access other books I mentioned to understand a concept better. 

* Foundational Exercises [[Jupyter notebooks]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/notebooks) 
* Toy problems and real-world problems [[Toy scripts]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/scripts) 
* A summary of the Book chapters can be accessed from subfolders [[theory folder]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/theory)
* A summary of my daily notes [[theory folder]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/theory)


Why going bayesian is good?
--------
The main advantages of going bayesian are as follows:
* Bayesian methods typically involves using probability distributions rather than point probabilities such as mean. For example for a regression problem for predicting house prices, rather than predicting the price of a house, bayesian methods produces a distribution of possible predictions. 
* Bayesian methods helps to derive credible intervals (similar to confidence interval but not the same) around the mean using the predicted distribution.
* Bayesian method can utilize informed or uninformed priors. Priors are nothing but prior knowledge about the distribution of samples. This is extremely useful fot getting better predictions as well as decreasing time required for traing a ML or DL model.
* Bayesian methods work efficiently even with small sample sizes for deep learning models or machine learning models.
* Bayesian methods account for variability in the measurement of the data.
* Bayesian methods are a solution to the over-fitting problem. Bayesian approach allows us to set certain priors on our features. For ex: Bayesian regression with a normal prior is same as ridge regression(L2 regularisation)![Video](https://www.coursera.org/learn/bayesian-methods-in-machine-learning/lecture/p1FM9/linear-regression).


Books
--------

* Think Bayes
* Statistical rethinking
* Bayesian analysis with python
* Bayesian data analysis
* Information Theory, Inference, and Learning Algorithms
* Gaussian process for machine learning
* pattern recognition and machine learning
* Probability theory: The logic of science



Sequence of jupyter notebooks that I used to study bayesian inference by compiling different resources
--------
<table class="tg">
  <tr>
    <th class="tg-yw4l"><b>Name</b></th>
    <th class="tg-yw4l"><b>Description</b></th>
    <th class="tg-yw4l"><b>Category</b></th>
    <th class="tg-yw41"><b>Level</b></th>
    <th class="tg-yw4l"><b>Link </b></th>
    <th class="tg-yw4l"><b>Blog </b></th>
    
  </tr>
  
  <tr>
    <td class="tg-yw4l">Understanding probability and posterior probability distribution</td>
    <td class="tg-yw4l">Introduction to probability, understanding joint and conditional probability, Bayes theorem and posterior distributions</td>
    <td class="tg-yw4l">Probability</td>
    <td class="tg-yw4l">Beginner</td>
    <td class="tg-yw4l"><a href="https://github.com/AllenDowney/ThinkBayes2/blob/master/code/chap02.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    
  </tr>
  
  
  <tr>
    <td class="tg-yw4l">Sampling and Variational Inference</td>
    <td class="tg-yw4l">An introduction to sampling and variational methods.</td>
    <td class="tg-yw4l">Sampling and approximation methods</td>
    <td class="tg-yw4l">Advanced</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/drive/1pgfT_sdoyNoYKxr_5SHYDm7yS1ANNI0e">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    
  </tr>
  
  <tr>
    <td class="tg-yw4l">Introduction to Pyro</td>
    <td class="tg-yw4l">Introduction to Pyro, a probabilistic programming package build on top of Pytorch.</td>
    <td class="tg-yw4l">Bayesian Inference</td>
    <td class="tg-yw4l">Advanced</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/drive/1FvNCbnu16evlCXyxxzNPf1uHUkVkDSFJ">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    
  </tr>

  <tr>
    <td class="tg-yw4l">MNIST with Pyro</td>
    <td class="tg-yw4l">MNIST with Pyro.</td>
    <td class="tg-yw4l">Neural Networks</td>
    <td class="tg-yw4l">Advanced</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/paraschopra/bayesian-neural-network-mnist/blob/master/bnn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    
  </tr> 
  
</table>


Pymc3
-----


pgmpy
----


fbprophet
-----



