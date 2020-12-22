import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from modules.networks import LinearGaussianTree, TriResNet
from modules.models import ColliderModel
from modules.distributions import NormalDistribution

def norm(x):
    return x/np.std(x)

num_samples = 20000
depth = 8
node = -1

d_eps = 20
tree = LinearGaussianTree(node_size=d_eps,depth=depth,in_scale=0.1,scale=0.15)

sm = (tree.sample(num_samples).detach().numpy())

#plt.scatter(norm(sm[:,0,0]), norm(sm[:,0,2000]), c="b", alpha=0.1)
#plt.scatter(norm(sm[:,0,0]), norm(sm[:,0,1000]), c="g", alpha=0.1)
#plt.scatter(norm(sm[:,0,0]), norm(sm[:,0,1]), c="r", alpha=0.1)

#plt.scatter(norm(sm[:,0,0]), norm(sm[:,0,20]), c="g", alpha=0.1)
corr = [np.corrcoef(sm[:,0,node], sm[:,0,k])[0,1] for k in range(sm.shape[2])]
plt.plot(corr)

join_link = lambda x,y: x + y
emission = lambda x: x
prior_model = ColliderModel(depth=depth, sigma=0.1, in_sigma=1., join_link=join_link,
                            transition_distribution=NormalDistribution(),eps_generator=tree)

smpl,_,_,_ = prior_model.sample(num_samples)
smpl = torch.cat(smpl,1).detach().numpy()
corr = [np.corrcoef(smpl[:,node], smpl[:,k])[0,1] for k in range(smpl.shape[1])]
plt.plot(corr)

transformations = [TriResNet(d_x=1, d_epsilon=d_eps, epsilon_nu=0.001, in_pre_lambda=0.5, scale_w=0.4) for _ in range(2**depth-1)]
post_model = ColliderModel(depth=depth, sigma=0.1, in_sigma=1., join_link=join_link,
                           transition_distribution=NormalDistribution(),
                           transformations=transformations, eps_generator=tree)

smpl,_,_,_ = post_model.sample(num_samples)
smpl = torch.cat(smpl,1).detach().numpy()
corr = [np.corrcoef(smpl[:,node], smpl[:,k])[0,1] for k in range(smpl.shape[1])]
plt.plot(np.abs(corr))
plt.show()