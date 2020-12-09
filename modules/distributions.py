import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NormalDistribution():

    def __init__(self, scale=1.):
        self.scale = scale

    def log_prob(self, x, loc, scale=None):
        if scale is None:
            scale = self.scale
        dist = torch.distributions.normal.Normal(loc, scale)
        return dist.log_prob(x)

    def rsample(self, loc, scale=None, N=None):
        if scale is None:
            scale = self.scale
        dist = torch.distributions.normal.Normal(loc, scale)
        if N is not None:
            return dist.rsample((N,))
        else:
            return dist.rsample()

    def KL(self, loc1, scale1, loc2, scale2):
        dist1 = torch.distributions.normal.Normal(loc=loc1, scale=scale1)
        dist2 = torch.distributions.normal.Normal(loc=loc2, scale=scale2)
        return torch.distributions.kl.kl_divergence(dist1, dist2)


class BernoulliDistribution():

    def log_prob(self, x, logits):
        dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
        return dist.log_prob(x)

    def rsample(self, logits, N):
        dist = torch.distributions.bernoulli.Bernoulli(logits=logits)
        if N is not None:
            return dist.sample((N,))
        else:
            return dist.sample()

    def KL(self, logits1, logits2):
        dist1 = torch.distributions.bernoulli.Bernoulli(logits=logits1)
        dist2 = torch.distributions.bernoulli.Bernoulli(logits=logits2)
        return torch.distributions.kl.kl_divergence(dist1, dist2)


class CauchyDistribution():

    def log_prob(self, x, loc, scale):
        dist = torch.distributions.cauchy.Cauchy(loc, scale)
        return dist.log_prob(x)

    def rsample(self, loc, scale, N):
        dist = torch.distributions.cauchy.Cauchy(loc, scale)
        if N is not None:
            return dist.rsample((N,))
        else:
            return dist.rsample()

    def KL(self, loc1, scale1, loc2, scale2):
        dist1 = torch.distributions.cauchy.Cauchy(loc=loc1, scale=scale1)
        dist2 = torch.distributions.cauchy.Cauchy(loc=loc2, scale=scale2)
        return torch.distributions.kl.kl_divergence(dist1, dist2)


class StudentDistribution():

    def __init__(self, df):
        self.df = df

    def log_prob(self, x, loc, scale):
        dist = torch.distributions.studentT.StudentT(df=self.df, loc=loc, scale=scale)
        return dist.log_prob(x)

    def rsample(self, loc, scale, N):
        dist = torch.distributions.studentT.StudentT(df=self.df, loc=loc, scale=scale)
        if N is not None:
            return dist.rsample((N,))
        else:
            return dist.rsample()

    def KL(self, loc1, scale1, loc2, scale2):
        dist1 = torch.distributions.studentT.StudentT(df=self.df, loc=loc1, scale=scale1)
        dist2 = torch.distributions.studentT.StudentT(df=self.df, loc=loc2, scale=scale2)
        return torch.distributions.kl.kl_divergence(dist1, dist2)


class LaplaceDistribution():

    def log_prob(self, x, loc, scale):
        dist = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
        return dist.log_prob(x)

    def rsample(self, loc, scale, N):
        dist = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
        if N is not None:
            return dist.rsample((N,))
        else:
            return dist.rsample()

    def KL(self, loc1, scale1, loc2, scale2):
        dist1 = torch.distributions.laplace.Laplace(loc=loc1, scale=scale1)
        dist2 = torch.distributions.laplace.Laplace(loc=loc2, scale=scale2)
        return torch.distributions.kl.kl_divergence(dist1, dist2)