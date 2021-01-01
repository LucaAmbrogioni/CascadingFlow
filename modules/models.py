import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.networks import TriResNet


class ProbabilisticModel(nn.Module):

    def __init__(self):
        super(ProbabilisticModel, self).__init__()
        self.dist = None

    def _avg_gaussian_log_prob(self, xt, mu, sigma):
        try:
            sd_loss = torch.log(2 * np.pi * sigma ** 2)
        except TypeError:
            sd_loss = np.log(2 * np.pi * sigma ** 2)
        return torch.sum(torch.mean(- (xt - mu) ** 2 / (2 * sigma ** 2) - 0.5 * sd_loss, 0))

    def _avg_log_prob(self, xt, mu, sigma, dist=None):
        if dist is None:
            dist = self.dist
        return torch.sum(torch.mean(dist.log_prob(x=xt, loc=mu, scale=sigma), 0))

    def _avg_log_likelihood(self, xt, yt, scale=None, regressor=None):
        N = xt.shape[0]
        if yt.shape[0] == 1:
            yt = yt.repeat((N, 1))
        zt = self.emission(xt, regressor)
        if scale is None:
            return torch.mean(self.emission_distribution.log_prob(yt, zt))
        else:
            return torch.mean(self.emission_distribution.log_prob(yt, zt, scale))

    def sample_observations(self, N, scale=None):   #TODO: needs refactoring
        x, mu, _, _, _ = self.sample_timeseries(N)
        if scale is None:
            return x, self.emission_distribution.rsample(self.emission(x), None), mu
        else:
            return x, self.emission_distribution.rsample(self.emission(x), scale), mu

    def sample_hierarchical_observations(self, N, M):
        x, _, _, _ = self.sample(N)
        y = []
        for k, child in enumerate(x["children"]):
            r = self.regressors[k] if self.regressors is not None else None
            M = None if self.regressors is not None else M
            y.append(self.emission_distribution.rsample(self.emission(child, r), self.emission_sigma_list[k], M))
        return x, y

    def _estimate_kl(self, smpl1, smpl2, s=0.01):
        n = smpl1.shape[1]
        mean1 = torch.mean(smpl1, 0)
        cov1 = torch.matmul(torch.transpose(smpl1 - mean1, 1, 0), smpl1 - mean1) + s*torch.Tensor(np.identity(n))
        mean2 = torch.mean(smpl2, 0)
        cov2 = torch.matmul(torch.transpose(smpl2 - mean2, 1, 0), smpl2 - mean2) + s*torch.Tensor(np.identity(n))
        dist1 = torch.distributions.multivariate_normal.MultivariateNormal(mean1, covariance_matrix=cov1)
        dist2 = torch.distributions.multivariate_normal.MultivariateNormal(mean2, covariance_matrix=cov2)
        return torch.sum(torch.distributions.kl.kl_divergence(dist1, dist2))

    def cascading_transformation(self, x, tr_index, log_jacobian, epsilon_loss, eps_mean=None, eps_scale=None, local_eps=None):
        N = x.shape[0]
        m = eps_mean if eps_mean is not None else 0.
        scale = eps_scale if eps_scale is not None else self.transformations[0].epsilon_nu
        x, new_epsilon_in, new_epsilon_out, new_log_jacobian = self.transformations[tr_index](x.squeeze(), local_eps=local_eps)
        log_jacobian += new_log_jacobian
        epsilon_loss += -self._avg_gaussian_log_prob(new_epsilon_out, m, scale)
        epsilon_loss += self._avg_gaussian_log_prob(new_epsilon_in, m, scale)
        x = x.view((N, self.d_x, 1))
        #print(new_epsilon_out.detach().numpy())
        return x, log_jacobian, epsilon_loss

    def reshape_collider_samples(self, x, depth):
        samples = []
        n = 0
        for d in range(depth-1):
            samples.append(x[:,0,n:n+2**(depth-d-1)])
            n += 2**(depth-d-1)
        return samples


class ColliderModel(ProbabilisticModel):

    def __init__(self,depth, sigma, in_sigma, join_link, transition_distribution,
                 transformations=(), mu_transformations=(), eps_generator=None):
        super(ColliderModel, self).__init__()
        self.depth = depth
        self.sigma = sigma
        self.in_sigma = in_sigma
        self.join_link = join_link
        self.transition_distribution = transition_distribution
        self.eps_generator = eps_generator
        self.d_x = 1
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

    def sample(self, N):
        log_jac = 0.
        eps_loss = 0.

        samples_list = []
        samples_pre_list = []
        tot_idx = 0
        # Latent cascading flow variables
        if self.eps_generator is not None:
            local_eps, eps_mean, eps_scale = [x.type(torch.float32) for x in self.eps_generator.sample(N)]
        else:
            local_eps, eps_mean, eps_scale = None, None, None
        for d in range(self.depth-1):
            par_idx = 0
            samples_d_list = []
            samples_pre_d_list = []
            for var_idx in range(2**(self.depth-d - 1)):
                if d == 0:
                    m = torch.zeros((N,))
                else:
                    m = self.join_link(samples_list[d-1][:,par_idx], samples_list[d-1][:,par_idx+1])
                s = self.sigma if d > 0 else self.in_sigma
                par_idx += 2

                # Pseudo-conjugate update
                if self.is_mu_transformed:
                    m, s = self.mu_transformations[tot_idx](m, s)

                # Transition sampling
                new_x = self.transition_distribution.rsample(m,s).view(N,1)

                # Flow transformation
                if self.is_transformed:
                    new_x_tr, log_jac, eps_loss = self.cascading_transformation(new_x, tot_idx, log_jac, eps_loss,
                                                                                eps_mean=eps_mean[:, :, tot_idx],
                                                                                eps_scale=eps_scale[:, :, tot_idx],
                                                                                local_eps=local_eps[:, :, tot_idx])
                    new_x_tr = new_x_tr.view(new_x.shape)
                else:
                    new_x_tr = new_x

                samples_d_list.append(new_x_tr)
                samples_pre_d_list.append(new_x)
                tot_idx += 1
            samples_list.append(torch.cat(samples_d_list, 1))
            samples_pre_list.append(torch.cat(samples_pre_d_list, 1))
        return samples_list, samples_pre_list, log_jac, eps_loss

    def evaluate_avg_joint_log_prob(self, samples_list, y=None, samples_pre_list=None, log_jacobian=None, epsilon_loss=None):
        avg_log_prob = 0.
        if log_jacobian:
            avg_log_prob -= log_jacobian
        if epsilon_loss:
            avg_log_prob += epsilon_loss
        for d in range(self.depth):
            par_idx = 0
            for var_idx in range(2 ** (self.depth - d - 1)):
                if d < self.depth - 1 or y is not None:
                    if d == 0:
                        m = 0.
                    else:
                        m = self.join_link(samples_list[d - 1][:, par_idx], samples_list[d - 1][:, par_idx + 1])
                    s = self.sigma if d > 0 else self.in_sigma
                    if d == self.depth-1:
                        c = y
                    elif samples_pre_list is not None:
                        c = samples_pre_list[d][:,var_idx]
                    else:
                        c = samples_list[d][:,var_idx]
                    avg_log_prob += self._avg_log_prob(c, m, s, dist=self.transition_distribution)
                    par_idx += 2
        return avg_log_prob


class HierarchicalModel(ProbabilisticModel):

    def __init__(self,n_children, d_x, mean_sigma, scale_mu, scale_sigma, emission_sigma_list, mean_dist, scale_dist,
                 children_dist, mean_link, scale_link, emission, emission_distribution, regressors=None,
                 transformations=(), mu_transformations=()):
        self.n_children = n_children
        self.d_x=d_x
        self.mean_sigma = mean_sigma
        self.scale_mu = scale_mu
        self.scale_sigma = scale_sigma
        self.emission_sigma_list = emission_sigma_list
        self.mean_dist = mean_dist
        self.scale_dist = scale_dist
        self.children_dist = children_dist
        self.mean_link = mean_link
        self.scale_link = scale_link
        self.emission = emission
        self.emission_distribution = emission_distribution
        self.regressors = regressors
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

    def sample(self, N):

        # Output variables
        log_jacobian = 0.
        epsilon_loss = 0.

        # Initialize dynamic variables
        mean_m = 0.
        mean_s = self.mean_sigma
        scale_m = self.scale_mu
        scale_s = self.scale_sigma

        # Global cascading flow variable
        if self.is_transformed:
            global_nu = self.transformations[0].epsilon_nu
            global_d_eps = self.transformations[0].d_epsilon
            global_epsilon = torch.distributions.normal.Normal(0., global_nu).rsample((N, global_d_eps))

        # Pseudo-conjugate update
        if self.is_mu_transformed:
            mean_m, mean_s = self.mu_transformations[0](mean_m, mean_s)
            scale_m, scale_s = self.mu_transformations[1](mean_m, mean_s)

        # Sampling
        mean_variable_pre = self.mean_dist.rsample(mean_m, mean_s, N * self.d_x).view((N, self.d_x, 1))
        scale_variable_pre = self.mean_dist.rsample(scale_m, scale_s, N * self.d_x).view((N, self.d_x, 1))

        # Cascading transformation
        if self.is_transformed:
            mean_variable, log_jacobian, epsilon_loss = self.cascading_transformation(mean_variable_pre, 0,
                                                                                      log_jacobian,
                                                                                      epsilon_loss, global_epsilon) #TODO: Broken
            scale_variable, log_jacobian, epsilon_loss = self.cascading_transformation(scale_variable_pre, 1,
                                                                                       log_jacobian,
                                                                                       epsilon_loss, global_epsilon)
        else:
            mean_variable = mean_variable_pre
            scale_variable = scale_variable_pre

        children_sample_pre_list = []
        children_sample_list = []
        for child_index in range(self.n_children):
            child_m = self.mean_link(mean_variable)
            child_s = self.scale_link(scale_variable)

            # Pseudo-conjugate update
            if self.is_mu_transformed:
                child_m, child_s = self.mu_transformations[child_index + 2](child_m, child_s)

            # Sampling
            child_variable_pre = self.children_dist.rsample(child_m, child_s).view((N, self.d_x, 1))

            # Cascading transformation
            if self.is_transformed:
                child_variable, log_jacobian, epsilon_loss = self.cascading_transformation(child_variable_pre,
                                                                                           child_index+2,
                                                                                           log_jacobian,
                                                                                           epsilon_loss,
                                                                                           global_epsilon)
            else:
                child_variable = child_variable_pre

            children_sample_list.append(child_variable)
            children_sample_pre_list.append(child_variable_pre)
        x = {"mean": mean_variable, "scale": scale_variable, "children": children_sample_list}
        x_pre = {"mean": mean_variable_pre, "scale": scale_variable_pre, "children": children_sample_pre_list}
        return x, x_pre, log_jacobian, epsilon_loss


    def evaluate_avg_joint_log_prob(self, x, y=None, x_pre=None, log_jacobian=None, epsilon_loss=None):
        mean_m = 0.
        mean_s = self.mean_sigma
        scale_m = self.scale_mu
        scale_s = self.scale_sigma

        # Pseudo-conjugate update
        if self.is_mu_transformed:
            mean_m, mean_s = self.mu_transformations[0](mean_m, mean_s)
            scale_m, scale_s = self.mu_transformations[1](mean_m, mean_s)

        avg_log_prob = self._avg_log_prob(x["mean"] if x_pre is None else x_pre["mean"],
                                          mean_m, mean_s, dist=self.mean_dist)
        avg_log_prob += self._avg_log_prob(x["scale"] if x_pre is None else x_pre["scale"],
                                           scale_m, scale_s, dist=self.scale_dist)
        if log_jacobian:
            avg_log_prob -= log_jacobian
        if epsilon_loss:
            avg_log_prob += epsilon_loss

        for child_index in range(self.n_children):
            mu = self.mean_link(x["mean"])
            s = self.scale_link(x["scale"])
            if self.is_mu_transformed:
                mu, s = self.mu_transformations[child_index + 2](mu, s)
            c = x_pre["children"][child_index] if x_pre is not None else x["children"][child_index]
            avg_log_prob += self._avg_log_prob(c, mu, s, dist=self.children_dist)
            if y is not None:
                y_c = y[child_index]
                r = self.regressors[child_index] if self.regressors is not None else None
                avg_log_prob += self._avg_log_likelihood(x["children"][child_index], y_c,
                                                         self.emission_sigma_list[child_index],
                                                         r)
        return avg_log_prob


class DynamicModel(ProbabilisticModel):

    def __init__(self, sigma, T, initial_sigma, distribution, d_x, transition, emission, emission_distribution,
                 mu_sd=0.001,
                 observation_gain=1., transformations=(), mu_transformations=(), initial_mean=0., eps_generator=()):
        self.sigma = sigma
        self.d_x = d_x
        self.transition = transition
        self.emission = emission
        self.emission_distribution = emission_distribution
        self.mu_sd = mu_sd
        self.initial_sigma = initial_sigma
        self.initial_mean = initial_mean
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
        if eps_generator:
            self.eps_generator = eps_generator
            self.has_eps_generator = True
        else:
            self.has_eps_generator = False

    def sample_timeseries(self, N, data=None):
        # Output variables
        log_jacobian = 0.
        epsilon_loss = 0.

        # Auxiliary variables
        if self.has_eps_generator:
            eps_global = self.eps_generator.sample(N, data)
        else:
            eps_global = [0. for _ in range(self.T)]

        # Initialize dynamic variables
        mu = torch.distributions.normal.Normal(0., self.mu_sd).rsample((N,))
        mut = self.initial_mean
        st = self.initial_sigma
        if self.is_mu_transformed:
            mut, st = self.mu_transformations[0](mut, st)
        x_pre = self.dist.rsample(mut, st, N * self.d_x).view((N, self.d_x, 1))

        # Transform prior
        if self.is_transformed:
            #x = self.cascading_transformation(x_pre.squeeze(), 0, log_jacobian, epsilon_loss,
            #                                         global_epsilon=eps_global)
            x, new_epsilon_in, new_epsilon_out, new_log_jacobian = self.transformations[0](x_pre.squeeze())
            log_jacobian += new_log_jacobian
            eps_sigma = self.transformations[0].epsilon_nu
            epsilon_loss += - self._avg_gaussian_log_prob(new_epsilon_out, 0., eps_sigma) + self._avg_gaussian_log_prob(
                new_epsilon_in, 0., eps_sigma)
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
            else:
                new_x_tr = new_x

            # Timeseries concatenation
            x_pre = torch.cat((x_pre, new_x), 2)
            x = torch.cat((x, new_x_tr), 2)
        return x, mu, x_pre, log_jacobian, epsilon_loss

    def evaluate_avg_joint_log_prob(self, x, y, mu, x_pre=None, log_jacobian=None, epsilon_loss=None):
        avg_log_prob = self._avg_gaussian_log_prob(mu, 0., self.mu_sd)
        if log_jacobian:
            avg_log_prob -= log_jacobian
        if epsilon_loss:
            avg_log_prob += epsilon_loss
        avg_log_prob += self._avg_log_prob(x[:, :, 0], self.initial_mean, self.initial_sigma)
        for t in range(self.T):
            if t < self.T - 1:
                old_xt = x[:, :, t]
                mut = self.transition(x[:, :, t], mu)
                st = self.sigma
                if self.is_mu_transformed:
                    mut, st = self.mu_transformations[t](mut, st)
                xt = x_pre[:, :, t + 1] if x_pre is not None else x[:, :, t + 1]
                avg_log_prob += self._avg_log_prob(xt, mut, st)
            if y is not None and y.shape[1] > t:
                yt = y[:, t]
                avg_log_prob += self._avg_log_likelihood(x[:, :, t], yt)
        return avg_log_prob


class MeanField(ProbabilisticModel):

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
      avg_log_prob += self._avg_gaussian_log_prob(xt, mut, st)

    if y is not None:
         raise ValueError("You cannot evaluate data likelihood with a variational model")
    return avg_log_prob


class MultivariateNormal(ProbabilisticModel): #TODO: Work in progress

  def __init__(self, T, d_x, s = 0.01):
    super(MultivariateNormal, self).__init__()
    self.d_x = d_x
    self.T = T
    self.s = s
    self.means = nn.Parameter(torch.Tensor(np.random.normal(0, 0.2, (T*d_x + 1,))))
    self.pre_tril = nn.Parameter(torch.Tensor(np.zeros((T*d_x + 1, T*d_x + 1))))
    self.pre_diag = nn.Parameter(torch.Tensor(np.random.normal(0., 0.2, (T*d_x + 1,))))

  def sample_timeseries(self, N):
    scale_tril = torch.tril(self.pre_tril,-1) + torch.diag(F.softplus(self.pre_diag) + self.s)
    X = torch.distributions.multivariate_normal.MultivariateNormal(loc=self.means,
                                                                   scale_tril=scale_tril).rsample((N,))
    mu = X[:,0].view((N,1))
    x_pre = X[:, 1:].view((N,self.d_x,self.T))
    x = x_pre
    return x, mu, x_pre, 0., 0.

  def evaluate_avg_joint_log_prob(self, x, y, mu, x_pre=None, log_jacobian = None, epsilon_loss=None):
    n, a, b = x.shape
    X = torch.cat((mu, x.view((n,a*b))), 1)
    scale_tril = torch.tril(self.pre_tril,-1) + torch.diag(F.softplus(self.pre_diag) + self.s)
    avg_log_prob = torch.mean(torch.distributions.multivariate_normal.MultivariateNormal(loc=self.means,
                                                                                         scale_tril=scale_tril).log_prob(X))
    if y is not None:
        raise ValueError("You cannot evaluate data likelihood with a variational model")
    return avg_log_prob #TODO: work in progress


class GlobalFlow(ProbabilisticModel):

  def __init__(self, T, d_x, d_eps, mu_sd = 0.001, residual=False):
    super(GlobalFlow, self).__init__()
    self.mu_sd = mu_sd
    self.d_x = d_x
    self.d_eps = d_eps
    self.T = T
    self.transformation = TriResNet(d_x=self.T*self.d_x, d_epsilon=d_eps,
                                    epsilon_nu=0.1, in_pre_lambda= 3. if residual else None)

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

    if y is not None:
        raise ValueError("You cannot evaluate data likelihood with a variational model")
    return avg_log_prob


class Autoregressive(ProbabilisticModel):

  def __init__(self, T, d_x, transition_models, mu_sd=0.001):
    super(Autoregressive, self).__init__()
    self.mu_sd = mu_sd
    self.T = T
    self.d_x = d_x
    self.transition_models = transition_models
    self.initial_mean = nn.Parameter(torch.Tensor(np.random.normal(0., 0.1, (d_x,))))
    self.pre_deviations = nn.Parameter(torch.Tensor(np.random.normal(0.,1.,(T, d_x))))

  def sample_timeseries(self, N):
    mu = torch.distributions.normal.Normal(0., self.mu_sd).rsample((N,))
    x = torch.distributions.normal.Normal(self.initial_mean,
                                          F.softplus(self.pre_deviations[0,:])).rsample((N,)).view((N,self.d_x,1))
    x_pre = x
    for t in range(self.T-1):
        mut = self.transition_models[t](x[:,:,-1])
        sigmat = F.softplus(self.pre_deviations[t,:]).repeat((N,1))
        new_x = torch.distributions.normal.Normal(mut, sigmat).rsample().view((N,self.d_x,1))

        # Timeseries concatenation
        x_pre = torch.cat((x_pre, new_x), 2)
        x = torch.cat((x, new_x), 2)

    return x, mu, x_pre, 0., 0.

  def evaluate_avg_joint_log_prob(self, x, y, mu, x_pre=None, log_jacobian = None, epsilon_loss=None):
    avg_log_prob = self._avg_gaussian_log_prob(mu, 0., self.mu_sd*torch.ones((1,)))

    for t in range(self.T):
      mut = self.transition_models[t-1](x[:,:,t-1]) if t > 1 else self.initial_mean
      sigmat = F.softplus(self.pre_deviations[t, :])
      xt = x[:,:,t]
      avg_log_prob += self._avg_gaussian_log_prob(xt, mut, sigmat)

    if y is not None:
         raise ValueError("You cannot evaluate data likelihood with a variational model")
    return avg_log_prob