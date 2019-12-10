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