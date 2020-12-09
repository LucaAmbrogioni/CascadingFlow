import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeSeries(nn.Module):

    def __init__(self):
        super(TimeSeries, self).__init__()
        self.dist = None

    def _avg_gaussian_log_prob(self, xt, mu, sigma):
        try:
            sd_loss = torch.log(2 * np.pi * sigma ** 2)
        except TypeError:
            sd_loss = np.log(2 * np.pi * sigma ** 2)
        return torch.sum(torch.mean(-(xt - mu) ** 2 / (2 * sigma ** 2) - 0.5 * sd_loss, 0))

    def _avg_log_prob(self, xt, mu, sigma):
        return torch.sum(torch.mean(self.dist.log_prob(x=xt, loc=mu, scale=sigma), 0))

    def _avg_logistig_log_likelihood(self, xt, yt):
        N = xt.shape[0]
        if yt.shape[0] == 1:
            yt = yt.repeat((N, 1))
        zt = self.emission(xt)
        return torch.mean(self.emission_distribution.log_prob(yt, zt))

    def sample_observations(self, N):
        x, mu, _, _, _, _ = self.sample_timeseries(N)
        y = self.emission_distribution.rsample(self.emission(x), None)
        return x, y, mu

    def _estimate_kl(self, smpl1, smpl2):
        mean1 = torch.mean(smpl1, 0)
        cov1 = torch.matmul(torch.transpose(smpl1 - mean1, 1, 0), smpl1 - mean1)
        mean2 = torch.mean(smpl2, 0)
        cov2 = torch.matmul(torch.transpose(smpl2 - mean2, 1, 0), smpl2 - mean2)
        dist1 = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=cov1)
        dist2 = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=cov2)
        return torch.sum(torch.distributions.kl.kl_divergence(dist1, dist2))


class DynamicModel(TimeSeries):

    def __init__(self, sigma, T, initial_sigma, distribution, d_x, transition, emission, emission_distribution,
                 mu_sd=0.001,
                 observation_gain=1., transformations=(), mu_transformations=()):
        self.sigma = sigma
        self.d_x = d_x
        self.transition = transition
        self.emission = emission
        self.emission_distribution = emission_distribution
        self.mu_sd = mu_sd
        self.initial_sigma = initial_sigma
        self.dist = distribution
        self.T = T
        self.observation_gain = observation_gain
        if transformations:
            self.transformations = transformations
            self.is_transformed = True
        else:
            self.is_transformed = False
        if mu_transformations:
            self.mu_transformations = mu_transformations
            self.is_mu_transformed = True
        else:
            self.is_mu_transformed = False

    def sample_timeseries(self, N):
        # Output variables
        log_jacobian = 0.
        epsilon_loss = 0.
        correction = 0.

        # Initialize dynamic variables
        mu = torch.distributions.normal.Normal(0., self.mu_sd).rsample((N,))
        mut = 0.
        st = self.initial_sigma
        if self.is_mu_transformed:
            mut, st = self.mu_transformations[0](mut, st)
        x_pre = self.dist.rsample(mut, st, N * self.d_x).view((N, self.d_x, 1))

        # Transform prior
        if self.is_transformed:
            x, new_epsilon_in, new_epsilon_out, new_log_jacobian = self.transformations[0](x_pre.squeeze())
            log_jacobian += new_log_jacobian
            eps_sigma = self.transformations[0].epsilon_nu
            epsilon_loss += - self._avg_gaussian_log_prob(new_epsilon_out, 0., eps_sigma) + self._avg_gaussian_log_prob(
                new_epsilon_in, 0., eps_sigma)
            correction += self._estimate_kl(new_epsilon_out, new_epsilon_in)
            x = x.view((N, self.d_x, 1))
        else:
            x = x_pre

        for t in range(self.T - 1):
            # Dynamic transition
            mut = self.transition(x[:, :, t], mu)
            st = self.sigma

            # Parameter transformation (ASVI)
            if self.is_mu_transformed:
                mut, st = self.mu_transformations[t](mut, st)
                new_x = self.dist.rsample(mut, st, None).view((N, self.d_x, 1))
            else:
                new_x = self.dist.rsample(mut, st, None).view((N, self.d_x, 1))

            # Flow transformation
            if self.is_transformed:
                new_x_tr, new_epsilon_in, new_epsilon_out, new_log_jacobian = self.transformations[t](new_x.squeeze())
                new_x_tr = new_x_tr.view((N, self.d_x, 1))
                log_jacobian += new_log_jacobian
                eps_sigma = self.transformations[t].epsilon_nu
                epsilon_loss += - self._avg_gaussian_log_prob(new_epsilon_out, 0.,
                                                              eps_sigma) + self._avg_gaussian_log_prob(new_epsilon_in,
                                                                                                       0., eps_sigma)
                correction += self._estimate_kl(new_epsilon_out, new_epsilon_in)
            else:
                new_x_tr = new_x

            # Timeseries concatenation
            x_pre = torch.cat((x_pre, new_x), 2)
            x = torch.cat((x, new_x_tr), 2)
        return x, mu, x_pre, log_jacobian, epsilon_loss, correction

    def evaluate_avg_joint_log_prob(self, x, y, mu, x_pre=None, log_jacobian=None, epsilon_loss=None, correction=None):
        avg_log_prob = self._avg_gaussian_log_prob(mu, 0., self.mu_sd)
        if log_jacobian:
            avg_log_prob -= log_jacobian
        if epsilon_loss:
            avg_log_prob += epsilon_loss
        if correction:
            avg_log_prob -= correction
        avg_log_prob += self._avg_log_prob(x[:, :, 0], 0., self.initial_sigma)
        for t in range(self.T):
            if t < self.T - 1:
                old_xt = x[:, :, t]
                mut = self.transition(x[:, :, t], mu)
                st = self.sigma
                if self.is_mu_transformed:
                    mut, st = self.mu_transformations[t](mut, st)
                xt = x_pre[:, :, t + 1] if x_pre is not None else x[:, :, t + 1]
                avg_log_prob += self._avg_log_prob(xt, mut, st)
            if y is not None:
                yt = y[:, t]
                avg_log_prob += self._avg_logistig_log_likelihood(x[:, :, t], yt)
        return avg_log_prob


class MeanField(TimeSeries):

  def __init__(self, T, d_x, mu_sd=0.001, transformations = ()):
    super(MeanField, self).__init__()
    self.mu_sd = mu_sd
    self.T = T
    self.d_x = d_x
    self.initial_means = nn.Parameter(torch.Tensor(np.random.normal(0,0.2,(T,d_x))))
    self.initial_pre_deviations = nn.Parameter(torch.Tensor(np.random.normal(0.1,0.1,(T,d_x))))
    if transformations:
      self.transformations = transformations
      self.is_transformed = True
    else:
      self.is_transformed = False

  def sample_timeseries(self, N):
    log_jacobian = 0.
    epsilon_loss = 0.
    mu = torch.distributions.normal.Normal(0., self.mu_sd).rsample((N,))
    x = torch.Tensor()
    for t in range(self.T):
      new_x = torch.distributions.normal.Normal(self.initial_means[t,:], F.softplus(self.initial_pre_deviations[t,:])).rsample((N,)).view((N,self.d_x,1))
      if self.is_transformed:
        new_x_tr, new_epsilon_in, new_epsilon_out, new_log_jacobian = self.transformations[t](new_x)
        log_jacobian += new_log_jacobian
        eps_sigma = self.transformations[t].epsilon_nu
        epsilon_loss += - self._avg_gaussian_log_prob(new_epsilon_out, 0., eps_sigma) + self._avg_gaussian_log_prob(new_epsilon_in, 0., eps_sigma)
      else:
        new_x_tr = new_x
      if t > 0:
        x_pre = torch.cat((x_pre, new_x), 2)
        x = torch.cat((x, new_x_tr), 2)
      else:
        x_pre = new_x
        x = new_x_tr
    return x, mu, x_pre, log_jacobian, epsilon_loss

  def evaluate_avg_joint_log_prob(self, x, y, mu, x_pre=None, log_jacobian = None, epsilon_loss=None):
    avg_log_prob = self._avg_gaussian_log_prob(mu, 0., self.mu_sd*torch.ones((1,)))
    if log_jacobian:
      avg_log_prob -= log_jacobian
    if epsilon_loss:
      avg_log_prob += epsilon_loss
    if x_pre is not None:
      x = x_pre
    for t in range(self.T):
      mut = self.initial_means[t]
      st = F.softplus(self.initial_pre_deviations[t])
      xt = x[:,:,t]
      avg_log_prob += self._avg_gaussian_log_prob(xt, mut, st) #TODO
      if y is not None:
        yt = y[:,t]
        avg_log_prob += self._avg_logistig_log_likelihood(xt, yt)
    return avg_log_prob


class GlobalFlow(TimeSeries):

  def __init__(self, T, d_x, d_eps, mu_sd = 0.001):
    super(GlobalFlow, self).__init__()
    self.mu_sd = mu_sd
    self.d_x = d_x
    self.d_eps = d_eps
    self.T = T
    self.transformation = TriResNet(d_x=self.T*self.d_x, d_epsilon=d_eps, epsilon_nu=0.1, in_pre_lambda=-5.)

  def sample_timeseries(self, N):
    mu = torch.distributions.normal.Normal(0., self.mu_sd).rsample((N,))
    x_pre = torch.distributions.normal.Normal(0., 1.).rsample((N,self.d_x*self.T))
    x, new_epsilon_in, new_epsilon_out, log_jacobian = self.transformation(x_pre)
    eps_sigma = self.transformation.epsilon_nu
    epsilon_loss = - self._avg_gaussian_log_prob(new_epsilon_out, 0., eps_sigma) + self._avg_gaussian_log_prob(new_epsilon_in, 0., eps_sigma)
    return x.view((N,self.d_x,self.T)), mu, x_pre, log_jacobian, epsilon_loss

  def evaluate_avg_joint_log_prob(self, x, y, mu, x_pre=None, log_jacobian = None, epsilon_loss=None):
    avg_log_prob = self._avg_gaussian_log_prob(mu, 0., self.mu_sd*torch.ones((1,)))
    if log_jacobian:
      avg_log_prob -= log_jacobian
    if epsilon_loss:
      avg_log_prob += epsilon_loss
    if x_pre is not None:
      x = x_pre
    avg_log_prob += self._avg_gaussian_log_prob(x, 0., 1.)
    for t in range(self.T):
      xt = x[:,t]
      if y is not None:
        yt = y[:,t]
        avg_log_prob += self._avg_logistig_log_likelihood(xt, yt)
    return avg_log_prob