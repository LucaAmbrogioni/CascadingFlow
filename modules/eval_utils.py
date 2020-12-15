import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def gauss_evaluate_likelihood(X, x_true):
  if len(X.shape) == 3:
    a,b,c = X.shape
    X = np.reshape(X, newshape=(a,b*c))
    x_true = np.reshape(x_true, newshape=(b*c,))
  X_mean = np.mean(X,0)
  X_sd = np.std(X,0)
  return np.mean(-(x_true - X_mean)**2/(2*X_sd**2) - 0.5*np.log(2*np.pi*X_sd**2))

def evaluate_likelihood(X, x_true, s=0.00001):
  def ker(x,y,bw):
      return np.mean(np.log(np.mean(np.exp(-(x - y)**2/(2*bw**2))/(np.sqrt(2*np.pi)*bw),0) + s))
  if len(X.shape) == 3:
    N,b,c = X.shape
    a = b*c
  else:
    N,a = X.shape
  X = np.reshape(X, newshape=(N,a))
  x_true = np.reshape(x_true, newshape=(1,a))
  bw = 0.9*np.std(X,0)*N**(-1/5)
  return ker(X, x_true, bw)

def evaluate_multi_likelihood(X, x_true, s=0.01):
  try:
    if len(X.shape) == 3:
      a,b,c = X.shape
      N = b*c
      X = np.reshape(X, newshape=(a,b*c))
      x_true = np.reshape(x_true, newshape=(b*c,))
    else:
      N = X.shape[1]
    mu = np.mean(X,0)
    S = np.cov(np.transpose(X)) + s*np.identity(N)
    return stats.multivariate_normal.logpdf(x_true, mean=mu, cov=S)
  except np.linalg.LinAlgError:
    print("Something went wrong in the multivariate evaluation metric")
    return np.nan

def evaluate_model(variational_model, X_true, M, emission_model, emission_distribution, scale, out_data, T_data):
  X, mu, x_pre, log_jacobian, epsilon_loss = variational_model.sample_timeseries(M)
  uni_lk = evaluate_likelihood(X.detach().numpy(), X_true.detach().numpy())
  multi_lk = evaluate_multi_likelihood(X.detach().numpy(), X_true.detach().numpy())
  multi_pred_lk = evaluate_predictive_error(X, emission_model, emission_distribution, scale, out_data, T_data)
  print("Avarage univariate latent likelihood: {}".format(uni_lk))
  print("Multivariate latent likelihood: {}".format(multi_lk))
  print("Predictive observable likelihood: {}".format(multi_pred_lk))
  return uni_lk, multi_lk, multi_pred_lk

def evaluate_predictive_error(X, emission_model, emission_distribution, scale, out_data, T_data):
  Y = np.zeros((X.shape[0], X.shape[2] - T_data))
  for t in range(T_data, X.shape[2]):
    tau = t - T_data
    eps = np.random.normal(0,0.0001,(Y.shape[0],))
    if scale is None:
      Y[:,tau] = emission_distribution.rsample(emission_model(X[:,:,t]), None).detach().numpy() + eps
    else:
      Y[:,tau] = emission_distribution.rsample(emission_model(X[:,:,t]), scale).detach().numpy() + eps
  #plt.plot(np.transpose(Y), alpha=0.01, c="r")
  #plt.plot(np.transpose(X[:20,0,T_data:].detach().numpy()), alpha=0.8, c="k")
  #plt.plot(np.transpose(out_data))
  #plt.show()
  return evaluate_likelihood(Y, out_data.detach().numpy())

