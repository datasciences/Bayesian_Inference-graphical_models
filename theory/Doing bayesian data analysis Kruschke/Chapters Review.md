Overview
--------


Chapter 1
---------
The book assumes no prior knowledge of statistics but basics of probabilitty and calculus. The author covers how to read this book chapter by chapter. He lays out a plan that is good for different knowledge levels and time limits. 
The contents of the chapters are given below as given in the text book:

Chapter 2: The idea of Bayesian inference and model parameters. This chapter
introduces important concepts; don’t skip it.
• Chapter 3: The R programming language. Read the sections about installing the
software, including the extensive set of programs that accompany this book. The rest
can be skimmed and returned to later when needed.
• Chapter 4: Basic ideas of probability. Merely skim this chapter if you have a high
probability of already knowing its content.
• Chapter 5: Bayes rule!
• Chapter 6: The simplest formal application of Bayes rule, referenced throughout the
remainder of the book.
4 Doing Bayesian Data Analysis
• Chapter 7: MCMC methods. This chapter explains the computing method that
makes contemporary Bayesian applications possible. You don’t need to study all the
mathematical details, but you should be sure to get the gist of the pictures.
• Chapter 8: The JAGS programming language for implementing MCMC.
• Chapter 16: Bayesian estimation of two groups. All of the foundational concepts from
the aforementioned chapters, applied to the case of comparing two groups.

Table 1 in chapter 1 also illustrates the Bayesian analogues of null hypothesis significance tests such as binomial tests, t-test etc.

There are links to R and Python implementation and workshops, however they do not work anymore, so don't get disappointed. You should be able to find the links on github, if you do a rigorous search. It is not relevant at this moment for me. I will add it here later, if I get a chance to implement the practicals. But for now I'm trying to understand the core concepts in detail.

Chapter 2
---------
Introduction: Credibility, Models, and Parameters

Chapter 2 goes through the basics of bayesian such as prior and posterior probabilites. You can skip this section if you already know these topics. I personally found the examples he used very simplistic and understandable, so I recommend reading it. 

The first take away from the first section of the chapter is "Bayesian Inference Is Reallocation of Credibility Across Possibilities". This means when you start solving a problem using bayesian methods, you start with a hypothesis. As you get more and more data or evidence you update the prior beliefs and you get the posterior beliefs. Since we are dealing with data, you say "Prior distribution" and "Posterior distribution". One of the advantage of using a distribution rather than a mean in bayesian methods is that, you account for variability in the measurement of the data. For example: You are measuring the blood pressure of several patients, the blood pressure of the same patient can vary minutely depending on several factors. So having a model that can account for this variability is always great.





















