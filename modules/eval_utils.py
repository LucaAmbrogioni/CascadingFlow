import numpy as np
import scipy.stats as stats


def evaluate_likelihood(X, x_true):
  if len(X.shape) == 3:
    a,b,c = X.shape
    X = np.reshape(X, newshape=(a,b*c))
    x_true = np.reshape(x_true, newshape=(b*c,))
  X_mean = np.mean(X,0)
  X_sd = np.std(X,0)
  return np.mean(-(x_true - X_mean)**2/(2*X_sd**2) - 0.5*np.log(2*np.pi*X_sd))


def evaluate_multi_likelihood(X, x_true):
  if len(X.shape) == 3:
    a,b,c = X.shape
    X = np.reshape(X, newshape=(a,b*c))
    x_true = np.reshape(x_true, newshape=(b*c,))
  mu = np.mean(X,0)
  S = np.cov(np.transpose(X))
  return stats.multivariate_normal.logpdf(x_true, mean=mu, cov=S)

