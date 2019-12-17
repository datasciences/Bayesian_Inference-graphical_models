#Load the data
data(cars)

#fit a linear regression
m <- lm( dist ~ speed , data=cars )
# estimated coefficients from the model coef(m)
# plot residuals against speed 
# Residual = observed y – predicted y
plot( resid(m) ~ speed , data=cars )

install.packages(c("coda","mvtnorm","devtools"))
library(devtools) 
devtools::install_github("rmcelreath/rethinking",ref="Experimental")


# exponential family of distributions
# normal,binomial,Poisson

 library(stats)
 par(mfrow=c(3,2))
 x=seq(0,1,by=0.1)
 alpha=c(0,2,10,20,50,500) # Trials
 beta=c(0,2,8,11,27,232) # Positive outcome
 for(i in 1:length(alpha)){
  y<-dbeta(x,shape1=alpha[i],shape2=beta[i])
  plot(x,y,type="l")
 }
 