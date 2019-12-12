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