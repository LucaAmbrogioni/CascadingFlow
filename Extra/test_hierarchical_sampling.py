import numpy as np
import matplotlib.pyplot as plt

import torch

from modules.distributions import NormalDistribution, BernoulliDistribution
from modules.models import HierarchicalModel
from modules.networks import TriResNet

mean_sigma = 1.
scale_mu = 1.
scale_sigma = 0.1
n_children = 10
emission_sigma_list = [0.5 for _ in range(n_children)]
d_x = 1
mean_dist = NormalDistribution()
scale_dist = NormalDistribution()
children_dist= NormalDistribution()
mean_link = lambda x: x
scale_link = lambda x: torch.exp(x)
emission = lambda x, r: x
emission_distribution = NormalDistribution()

model = HierarchicalModel(n_children, d_x, mean_sigma, scale_sigma, scale_mu, emission_sigma_list, mean_dist, scale_dist,
                          children_dist, mean_link, scale_link, emission, emission_distribution)
N = 40
M = 500
_, y  = model.sample_hierarchical_observations(N, M)

for n in range(n_children):
    plt.hist(y[n][:,0].detach().numpy().flatten(), 25, alpha=0.5)
plt.show()

transformations = [TriResNet(d_x=d_x, d_epsilon=10, epsilon_nu=0.1, in_pre_lambda=1.) for _ in range(2 + n_children)]

tr_model = HierarchicalModel(n_children, d_x, mean_sigma, scale_sigma, scale_mu, emission_sigma_list, mean_dist, scale_dist,
                             children_dist, mean_link, scale_link, emission, emission_distribution,
                             transformations=transformations)

_, tr_y  = tr_model.sample_hierarchical_observations(N, M)

for n in range(n_children):
    plt.hist(tr_y[n][:,0].detach().numpy().flatten(), 25, alpha=0.5)
plt.show()

#X_true, Y, mu =  bm.sample_observations(N)