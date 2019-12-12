#Load the data
data(cars)

#fit a linear regression
m <- lm( dist ~ speed , data=cars )
# estimated coefficients from the model coef(m)
# plot residuals against speed 
# Residual = observed y – predicted y
plot( resid(m) ~ speed , data=cars )
