# Bayesian machine learning and deep learning workout area with Brms, Pymc3, Pyro and Gpytorch

Uncertainity estimation in machine learning, deep learning and reinforcement learning methods are increasingly becoming popular due to the latest research advancements in variational inference and dropout methods. 

This repository will contain book chapter reviews, bayesian model implementations and resources for learning bayesian modeling. I did a rigorous research on this topic to come up with a list of most influential books and programming packages on this topic to layout a plan for my study. The list of books and packages are listed under Books. 

Overview 
--------
There will be several folders and subfolders in this repository. 

* A summary of the Book chapters can be accessed from subfolders [[theory folder]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/theory)
* A summary of my daily notes [[theory folder]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/theory)
* All basic statistical exercises from think bayes will go into [[stats folder]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/stats)
* All practical exercises using machine learning will go into folder [[ml]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/ml)
* All practical exercises using deep learning will go into folder [[dl]](https://github.com/vvrahul11/Bayesian_ml_dl_workout_area/tree/master/dl)
* Both ml and dl folders will contain subfolders with examples, inference from R and python libraries such as Pymc3, Brms, Pyro, gpytorch or Botorch. 
* ml using Pymc3
* dl using Pymc3


Why going bayesian is good?
--------
The main advantages of going bayesian are as follows:
* Bayesian methods typically involves using probability distributions rather than point probabilities such as mean. For example for a regression problem for predicting house prices, rather than predicting the price of a house, bayesian methods produces a distribution of possible predictions. 
* Bayesian methods helps to derive credible intervals (similar to confidence interval but not the same) around the mean using the predicted distribution.
* Bayesian method can utilize informed or uninformed priors. Priors are nothing but prior knowledge about the distribution of samples. This is extremely useful fot getting better predictions as well as decreasing time required for traing a ML or DL model.
* Bayesian methods work efficiently even with small sample sizes for deep learning models or machine learning models.
* Bayesian methods account for variability in the measurement of the data


Core architecture of a bayesian method
--------


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

For some basic calculus: http://tutorial.math.lamar.edu/Classes/CalcII/Probability.aspx

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