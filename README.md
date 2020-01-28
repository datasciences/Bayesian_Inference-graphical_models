# Bayesian machine learning and deep learning workout area with Brms, Pymc3, Pyro (and possibly Gpytorch in the future)

![GitHub Logo](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/blob/master/images/Bayesian.jpg)


This serves as a repository containing my Bayesian Inference learnings. Please do not get overwhelmed by the names of several packages written in the title. During my learning curve, due to the resources I used, I moved from brms through pymc3 to pyro. Initially, when I started out, I built models in brms (An R package). It was quite to easy to build and improve the model. But, eventually when it got harder to assess the outputs, underlying theory and need for bayesian python models, I switched to Pymc3, which has a strong open source community around it. Later, once I picked up the theory and started building models in pymc3, I switched myself to pyro and trying my best to get expertise in it. You can read more about bayesian inference and the organisation of my resources below. 

Why I made this repository?

Uncertainity estimation in machine learning, deep learning and reinforcement learning methods are increasingly becoming popular due to the latest research advancements in variational inference, monte carlo dropout, bayes by backpropagation and its application in convolutional and recurrent neural networks. 

This repository will contain book chapter reviews, bayesian model implementations and resources for learning bayesian modeling. I did a rigorous research on this topic to come up with a list of most influential books and programming packages on this topic to layout a plan for my study. The list of books and packages are listed under Books. 

Overview 
--------
There will be several folders and subfolders in this repository. Currently, this is a compilation of several knowledge that I learned from different resoures. If you would like to follow along, I would suggest you to start with Think Bayes book and then stick to Statistical Rethinking book. Further, you can access exercises from "notebooks folder". For book chapters review, I have really enjoyed reading from different authors since it helped me to understand and reason about the same concept multiple time. Currently, I am sticking to statistical rethinking book, but I do access other books I mentioned to understand a concept better. 

* Exercises [[Jupyter notebooks]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/notebooks) 
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


If you want to learn some statistical modeling and get acquinted with statistical problem solving start with this book "https://bookdown.org/roback/bookdown-bysh/" suggested by Statistical Rethinking Record.

For basic calculus: http://tutorial.math.lamar.edu/Classes/CalcII/Probability.aspx


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
    <td class="tg-yw4l">Understanding probability and posterior probability distribution</td>
    <td class="tg-yw4l">Probability</td>
    <td class="tg-yw4l">Beginner</td>
    <td class="tg-yw4l"><a href="https://github.com/AllenDowney/ThinkBayes2/blob/master/code/chap02.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    <td class="tg-yw4l"><a href="https://docs.google.com/presentation/d/1D1uT92LgG5Rp5zTWeJhSjZzxihL-7oU8Ddsk18uZCBI/edit?usp=sharing">read</a></td>
  </tr>
  
  
  <tr>
    <td class="tg-yw4l">Sampling and Variational Inference</td>
    <td class="tg-yw4l">Sampling and Variational Inference.</td>
    <td class="tg-yw4l">Machine learning and Deep Learning</td>
    <td class="tg-yw4l">Advanced</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/drive/1pgfT_sdoyNoYKxr_5SHYDm7yS1ANNI0e">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    <td class="tg-yw4l"><a href="https://docs.google.com/presentation/d/1Bb-M69vHNezFNsmJKcjSqpNokWxAdc7j-kFfkfPve-A/edit?usp=sharing">read</a></td>
  </tr>
  
  <tr>
    <td class="tg-yw4l">Introduction to Pyro</td>
    <td class="tg-yw4l">Introduction to Pyro.</td>
    <td class="tg-yw4l">Bayesian Inference</td>
    <td class="tg-yw4l">Advanced</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/drive/1FvNCbnu16evlCXyxxzNPf1uHUkVkDSFJ">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    <td class="tg-yw4l"><a href="https://docs.google.com/presentation/d/1Bb-M69vHNezFNsmJKcjSqpNokWxAdc7j-kFfkfPve-A/edit?usp=sharing">read</a></td>
  </tr>

  <tr>
    <td class="tg-yw4l">MNIST with Pyro</td>
    <td class="tg-yw4l">MNIST with Pyro.</td>
    <td class="tg-yw4l">Neural Networks</td>
    <td class="tg-yw4l">Advanced</td>
    <td class="tg-yw4l"><a href="https://colab.research.google.com/github/paraschopra/bayesian-neural-network-mnist/blob/master/bnn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" width = '400px' >
</a></td>
    <td class="tg-yw4l"><a href="https://docs.google.com/presentation/d/1Bb-M69vHNezFNsmJKcjSqpNokWxAdc7j-kFfkfPve-A/edit?usp=sharing">read</a></td>
  </tr> 
  
</table>



Papers
--------

* Inference from Simulations and Monitoring Convergence: https://www.mcmchandbook.net/HandbookChapter6.pdf
* Dropout as a bayesian approximation http://proceedings.mlr.press/v48/gal16.pdf
* What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? https://arxiv.org/pdf/1703.04977.pdf
* Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
https://arxiv.org/pdf/1705.07115.pdf (https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/)


Talks
--------
1. https://www.youtube.com/channel/UC_LBLWLfKk5rMKDOHoO7vPQ
2. https://www.youtube.com/watch?v=HNKlytVD1Zg&t=3836s
   Bayesian logistic regression, KL divergence, neural network code : https://github.com/chrisorm/pydata-2018/tree/master/notebooks 


Blog
--------
1. http://fastml.com/bayesian-machine-learning/
2. https://medium.com/panoramic/gaussian-processes-for-little-data-2501518964e4


Gaussian Processes 
-------------
Gaussian processes are a powerful tool in the machine learning toolbox.They allow us to make predictions about our data by incorporating prior knowledge. Their most obvious area of application is fitting a function to the data. This is called regression and is used, for example, in robotics or time series forecasting. But Gaussian processes are not limited to regression — they can also be extended to classification and clustering tasks. For a given set of training points, there are potentially infinitely many functions that fit the data. Gaussian processes offer an elegant solution to this problem by assigning a probability to each of these functions. The mean of this probability distribution then represents the most probable characterization of the data. Furthermore, using a probabilistic approach allows us to incorporate the confidence of the prediction into the regression result.  - https://distill.pub/2019/visual-exploration-gaussian-processes/

1. DEEP NEURAL NETWORKS AS GAUSSIAN PROCESSES: https://arxiv.org/pdf/1711.00165v3.pdf
2. Automatic Differentiation Variational Inference: https://arxiv.org/pdf/1603.00788.pdf
3. The Variational Gaussian Approximation Revisited : https://www.mitpressjournals.org/doi/full/10.1162/neco.2008.08-07-592
4. https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
5. Deep Neural Networks as Gaussian Processes  [PDF] Lee, J., Sohl-Dickstein, J., Pennington, J., Novak, R., Schoenholz, S. and Bahri, Y., 2018. International Conference on Learning Representations.
6. Deep Gaussian Processes  [PDF] Damianou, A. and Lawrence, N., 2013. Proceedings of the Sixteenth International Conference on Artificial Intelligence and Statistics, Vol 31, pp. 207--215. PMLR.
7. https://nbviewer.jupyter.org/github/adamian/adamian.github.io/blob/master/talks/Brown2016.ipynb
8. http://katbailey.github.io/post/gaussian-processes-for-dummies/
9. http://katbailey.github.io/post/from-both-sides-now-the-math-of-linear-regression/
10. https://sigopt.com/blog/intuition-behind-gaussian-processes
11. http://www.tmpl.fi/gp/
12. https://towardsdatascience.com/using-bayesian-modeling-to-improve-price-elasticity-accuracy-8748881d99ba
13. https://distill.pub/2019/visual-exploration-gaussian-processes/
14. https://ax.dev/docs/bayesopt.html#a-closer-look-at-gaussian-processes
15. A tutorial on Bayesian optimization using gaussian process: https://arxiv.org/pdf/1807.02811.pdf
16. https://ax.dev/docs/bayesopt



Link between Bayesian inference, Gaussian processes and deep learning
----------------------------------------------
1. Deep Neural Networks as Gaussian Processes  [PDF] Lee, J., Sohl-Dickstein, J., Pennington, J., Novak, R., Schoenholz, S. and Bahri, Y., 2018. International Conference on Learning Representations.
2. Deep Gaussian Processes  [PDF] Damianou, A. and Lawrence, N., 2013. Proceedings of the Sixteenth International Conference on Artificial Intelligence and Statistics, Vol 31, pp. 207--215. PMLR.


Further Reading [Ref: https://medium.com/@Petuum/intro-to-modern-bayesian-learning-and-probabilistic-programming-c61830df5c50]
---------------
The following are some recommended papers cited throughout this blog post, broken down into categories:
Scalable Bayesian inference algorithms:
1. Bayesian Learning via Stochastic Gradient Langevin Dynamics
2. Stochastic Gradient Hamiltonian Monte Carlo
3. Big Learning with Bayesian Methods
Parallel and distributed Bayesian inference algorithms:
4. Asymptotically Exact, Embarrassingly Parallel MCMC
5. Parallelizing MCMC via Weierstrass Sampler
6. Scalable and Robust Bayesian Inference via the Median Posterior
Variational approximations and amortized inference:
7. Variational Inference: A Review for Statisticians
8. Stochastic Variational Inference
9. Stochastic Backpropagation and Approximate Inference in Deep Generative Models
10. Auto-Encoding Variational Bayes
Deep Bayesian learning:
11. Deep Probabilistic Programming
12. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
13. Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference
Simulators:
14. Automatic Inference for Inverting Software Simulators via Probabilistic Programming
15. Improvements to Inference Compilation for Probabilistic Programming in Large-Scale Scientific Simulators
Visual graphics:
16. Approximate Bayesian Image Interpretation using Generative Probabilistic Graphics Programs
17. Picture: A Probabilistic Programming Language for Scene Perception
Universal probabilistic programming:
18. Venture: a higher-order probabilistic programming platform with programmable inference
19. A New Approach to Probabilistic Programming Inference
20. Inference Compilation and Universal Probabilistic Programming
Verification, testing, quality assurance:
21. Debugging probabilistic programs
22. Testing Probabilistic Programming Systems
Usability:
23. BayesDB: A probabilistic programming system for querying the probable implications of data
24. Probabilistic Programs as Spreadsheet Queries
25. Spreadsheet Probabilistic Programming
Ecosystem and modularity:
26. Tensorflow probability
27. Pyro


A collection of recent papers in Variational Inference
-----------------------------
1. https://github.com/otokonoko8/implicit-variational-inference



Other probabilistic libraries in Python
-------------------------------
1. ptstat: Probabilistic Programming and Statistical Inference in PyTorch
2. pyro: Deep universal probabilistic programming with Python and PyTorch http://pyro.ai
3. probtorch: Probabilistic Torch is library for deep generative models that extends PyTorch.
4. paysage: Unsupervised learning and generative models in python/pytorch.
5. pyvarinf: Python package facilitating the use of Bayesian Deep Learning methods with Variational Inference for PyTorch.
6. pyprob: A PyTorch-based library for probabilistic programming and inference compilation.
7. mia: A library for running membership inference attacks against ML models.
8. pro_gan_pytorch: ProGAN package implemented as an extension of PyTorch nn.Module.
9. botorch: Bayesian optimization in PyTorch