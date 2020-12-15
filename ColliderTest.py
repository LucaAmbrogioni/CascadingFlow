import numpy as np
import matplotlib.pyplot as plt

import torch

from modules.networks import LinearGaussianTree
from modules.models import ColliderModel
from modules.distributions import NormalDistribution

tree = LinearGaussianTree(node_size=3,depth=5,in_scale=1.,scale=0.1)

sm = tree.sample(5000).detach().numpy()

join_link = lambda x,y: x + y
emission = lambda x: x
prior_model = ColliderModel(depth=5, sigma=0.1, join_link=join_link,
                            transition_distribution=NormalDistribution(),
                            emission=emission, emission_distribution=NormalDistribution(scale=0.01))

num_samples = 1000
smpl = prior_model.sample(num_samples).detach().numpy()
plt.plot(np.std(smpl,0))
plt.show()
plt.scatter(smpl[:,10], smpl[:,60])
plt.xlim(-1.2,1.2)
plt.ylim(-1.2,1.2)
plt.show()