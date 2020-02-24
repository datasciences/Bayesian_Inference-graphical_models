'''
Vriance and Covariance

Cov(X, Y) = sum(Xi - Xmean) (Yi - Ymean) / n-1


https://people.duke.edu/~ccc14/sta-663/PCASolutions.html#eigendecomposition-of-the-covariance-matrix
'''

def cov(X, Y):
  Xmean = np.mean(X)
  Ymean = np.mean(Y)
  cov  = np.sum((X - Xmean) * (Y - Ymean)/(len(X) -1))
  return cov

X = np.random.random(10)
Y = np.random.random(10)

# Find variance of X and Y
cov(X, Y)

# Create a covariance matrix of X and Y
np.array([[cov(X, X), cov(X, Y)], [cov(Y, X), cov(Y, Y)]])



'''
Note that it works only for square matrix
'''
