import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch.optim as optim

from modules.models import GlobalFlow, DynamicModel
from modules.training_tools import variational_update
from modules.networks import InferenceNet
from modules.dynamics import VolterraTransition

from modules.distributions import NormalDistribution
from modules.emissions import SingleCoordinateEmission

# Simulation parameters
num_iterations = 3000
batch_size = 100

# Model
T = 100
dt = 0.5
sigma = np.sqrt(dt)*0.5
initial_sigma = 3.
initial_mean = 0.
d_x = 2 #Number of latent variables
d_eps = 10
dist = NormalDistribution()
lk_sigma = 3.
#transition_model = lambda x,m: x #
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
inference_net = InferenceNet(in_size=T, out_size=T*d_x, n_hidden=T, out_shape=(T,d_x), eps=0.1)
variational_model = GlobalFlow(T,d_x, d_eps=10, inference_net=inference_net)

loss_list = []
params_list = [inference_net.parameters()] + [variational_model.transformation.parameters()]
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
    loss = variational_update(prior_model, variational_model, data, optimizer, batch_size, amortized=True)

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