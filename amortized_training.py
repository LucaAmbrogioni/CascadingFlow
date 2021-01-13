import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch.optim as optim

from modules.models import DynamicModel
from modules.training_tools import variational_update
from modules.networks import TriResNet, LinearGaussianChain
from modules.dynamics import VolterraTransition

from modules.distributions import NormalDistribution
from modules.emissions import SingleCoordinateEmission

# Simulation parameters
num_iterations = 3500
batch_size = 200

# Model
T = 50
dt = 0.5
sigma = np.sqrt(dt)*0.5
initial_sigma = 3.
initial_mean = 0.
d_x = 2 #Number of latent variables
d_eps = 10
dist = NormalDistribution()
lk_sigma = 0.1
transition_model = VolterraTransition(dt=dt)

# Likelihood
observation_gain = 1.
emission_model = SingleCoordinateEmission(k=0, gain=observation_gain)
emission_dist = NormalDistribution(scale=lk_sigma)

### Prior model ###
prior_model = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                           transition=transition_model,
                           emission=emission_model,
                           emission_distribution=emission_dist,
                           observation_gain=observation_gain, T=T, initial_mean=initial_mean)

### Cascading flow ###
print("Train cascading flows")
transformations = [TriResNet(d_x=d_x, d_epsilon=d_eps, epsilon_nu=0.1, in_pre_lambda=4.) for _ in range(T)]
auxiliary_model = LinearGaussianChain(node_size=d_eps, T=T, in_scale=0.1, amortized=True, data_dim=1)
variational_model = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                                 transition=transition_model,
                                 emission=emission_model, emission_distribution=emission_dist,
                                 observation_gain=observation_gain, T=T,
                                 transformations=transformations, initial_mean=initial_mean,
                                 eps_generator=auxiliary_model,
                                 is_amortized=True)

loss_list = []
params_list = [list(tr.parameters()) for tr in transformations] + [list(auxiliary_model.parameters())]
params = []
for p in params_list:
    params += p
optimizer = optim.Adam(params, lr=0.001)

# generate ground truth
#X_true, Y, mu = prior_model.sample_observations(batch_size)
#data = Y.view((batch_size,1,T))

# generate ground truth
#X_true, Y, mu = prior_model.sample_observations(1)
#data = Y[0, :].view((1, T))

for itr in tqdm(range(num_iterations)):

    # generate ground truth
    X_true, Y, mu = prior_model.sample_observations(batch_size)
    data = Y #[0, :].view((1, T))

    # Variational update
    loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)

    # Loss
    loss_list.append(float(loss.detach().numpy()))

## Plot results ##
plt.plot(loss_list)
plt.show()

# generate ground truth
M = 100
for _ in range(4):
    #X_true, Y, mu = prior_model.sample_observations(1)
    #data = Y.view((1,1,T)).repeat((M,1,1))
    #data = Y.view((1, 1, T)).repeat((M, 1, 1))
    # generate ground truth
    X_true, Y, mu = prior_model.sample_observations(1)
    data = Y[0, :].view((1,T)).repeat((M, 1))

    X, _, _, _, _ = variational_model.sample_timeseries(M, data)

    x = X.detach().numpy()[:,0,:]
    x_tr = X_true.detach().numpy()[0,0,:]
    #y = data.detach().numpy()[:, 0, :]

    #t_range = np.tile(np.linspace(0.,T*dt, T), (M,1))
    plt.plot(np.transpose(x), alpha=0.5)
    plt.plot(np.transpose(x_tr), c="k", lw=2, ls="--")
    #plt.scatter(t_range, np.transpose(y), c="b", lw=2)
    plt.show()