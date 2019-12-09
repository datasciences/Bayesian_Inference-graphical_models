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