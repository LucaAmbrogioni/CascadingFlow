import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch.optim as optim

from modules.models import DynamicModel, GlobalFlow, MeanField
from modules.training_tools import variational_update
from modules.networks import TriResNet, LinearGaussianChain, InferenceNet
from modules.dynamics import VolterraTransition, LorentzTransition, VanDerPollTransition, RNNTransition, BrusselatorTransition
from modules.eval_utils import evaluate_likelihood

from modules.distributions import NormalDistribution, BernoulliDistribution
from modules.emissions import SingleCoordinateEmission

# Defining dynamical and emission model
model_name = "bm"
lik_name = "r"
exp_name = model_name + "_" + lik_name
print(exp_name)

if model_name == "lz":
    T = 40
    T_data = 20
    dt = 0.02
    sigma = np.sqrt(dt)*2.
    initial_sigma = 1.
    initial_mean = 0.
    d_x = 3 #Number of latent variables
    dist = NormalDistribution()
    lk_sigma = 3.
    transition_model = LorentzTransition(dt=dt)
elif model_name == "vdp":
    T = 200
    T_data = 50
    dt = 0.005
    sigma = np.sqrt(dt) * 0.1
    initial_sigma = 2.
    initial_mean = 0.
    d_x = 2  # Number of latent variables
    dist = NormalDistribution()
    lk_sigma = 3.
    transition_model = VanDerPollTransition(dt=dt, mu=7., omega=20*np.pi*2)
elif model_name == "vol":
    T = 100
    T_data = 50
    dt = 0.5
    sigma = np.sqrt(dt)*0.5
    initial_sigma = 3.
    initial_mean = 0.
    d_x = 2 #Number of latent variables
    dist = NormalDistribution()
    lk_sigma = 3.
    transition_model = VolterraTransition(dt=dt)
elif model_name == "bruss":
    T = 100
    T_data = 50
    dt = 0.05
    sigma = np.sqrt(dt)*0.05
    initial_sigma = 4.
    initial_mean = 0.
    d_x = 2 #Number of latent variables
    dist = NormalDistribution()
    lk_sigma = 3.
    transition_model = BrusselatorTransition(dt=dt)
elif model_name == "bm":
    T = 40
    T_data = 20
    dt = 0.2
    sigma = np.sqrt(dt) * 1.
    initial_sigma = 1.
    initial_mean = 0.
    d_x = 1  # Number of latent variables
    dist = NormalDistribution()
    lk_sigma = 1.
    transition_model = lambda x,m: x
elif model_name == "rnn":
    T = 40  # 60
    T_data = 20
    dt = 0.2
    sigma = np.sqrt(dt) * 0.1  # 0.5
    initial_sigma = 1.
    initial_mean = 0.
    d_x = 3
    dist = NormalDistribution()
    lk_sigma = 1.
    transition_model = RNNTransition(d_x=d_x, d_h=5, dt=dt)

if lik_name == "c":
    observation_gain = 2.
    lk_sigma = None
    emission_model = SingleCoordinateEmission(k=0, gain=observation_gain)
    emission_dist = BernoulliDistribution()
elif lik_name == "r":
    observation_gain = 1.
    emission_model = SingleCoordinateEmission(k=0, gain=observation_gain)
    emission_dist = NormalDistribution(scale=lk_sigma)

# Simulation parameters
num_iterations = 800 #1000
batch_size = 200

# # Model
# T = 100
# dt = 0.5
# sigma = np.sqrt(dt)*0.5
# initial_sigma = 3.
# initial_mean = 0.
# d_x = 2 #Number of latent variables
# d_eps = 10
# dist = NormalDistribution()
# lk_sigma = 3.
# transition_model = VolterraTransition(dt=dt)
#
# # Likelihood
# observation_gain = 1.
# emission_model = SingleCoordinateEmission(k=0, gain=observation_gain)
# emission_dist = NormalDistribution(scale=lk_sigma)

### Prior model ###
prior_model = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                           transition=transition_model,
                           emission=emission_model,
                           emission_distribution=emission_dist,
                           observation_gain=observation_gain, T=T, initial_mean=initial_mean)

### Cascading flow ###
print("Train cascading flows")
d_eps = 10
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

for itr in tqdm(range(num_iterations)):

    # generate ground truth
    X_true, Y, mu = prior_model.sample_observations(batch_size)
    data = Y #[0, :].view((1, T))

    # Variational update
    loss = variational_update(prior_model, variational_model, data, optimizer, batch_size, amortized=True)

    # Loss
    loss_list.append(float(loss.detach().numpy()))

## Plot results ##
#plt.plot(loss_list)
#plt.show()

# generate ground truth
num_repetitions = 50
M = 100
lk_list = []
for _ in range(num_repetitions):
    X_true, Y, mu = prior_model.sample_observations(1)
    data = Y[0, :].view((1,T)).repeat((M, 1))

    X, _, _, _, _ = variational_model.sample_timeseries(M, data)

    lk_list.append(evaluate_likelihood(X.detach().numpy(), X_true.detach().numpy()))

    #x = X.detach().numpy()[:,0,:]
    #x_tr = X_true.detach().numpy()[0,0,:]
    #y = data.detach().numpy()[:, 0, :]

    t_range = np.tile(np.linspace(0.,T*dt, T), (M,1))
    plt.plot(np.transpose(x), alpha=0.5)
    plt.plot(np.transpose(x_tr), c="k", lw=2, ls="--")
    plt.scatter(t_range, np.transpose(y), c="b", lw=2)
    plt.show()

### Mean field ###
print("Train mean field")
inference_net = InferenceNet(in_size=T, out_size=T*d_x, n_hidden=T, out_shape=(T,d_x), eps=0.1)
variational_model = MeanField(T,d_x, inference_net=inference_net)

loss_list = []
params_list = [inference_net.parameters()]
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

lk_mf_list = []
for _ in range(num_repetitions):
    X_true, Y, mu = prior_model.sample_observations(1)
    data = Y[0, :].view((1,T)).repeat((M, 1))

    X, _, _, _, _ = variational_model.sample_timeseries(M, data)

    lk_mf_list.append(evaluate_likelihood(X.detach().numpy(), X_true.detach().numpy()))

### Cascading flow ###
print("Train global flow")
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


lk_gf_list = []
for _ in range(num_repetitions):
    X_true, Y, mu = prior_model.sample_observations(1)
    data = Y[0, :].view((1,T)).repeat((M, 1))

    X, _, _, _, _ = variational_model.sample_timeseries(M, data)

    lk_gf_list.append(evaluate_likelihood(X.detach().numpy(), X_true.detach().numpy()))


## Print metrics
print("Mean CF likelihood: {} += {}".format(np.mean(lk_list), np.std(lk_list)/np.sqrt(num_repetitions)))
print("Mean MF likelihood: {} += {}".format(np.mean(lk_mf_list), np.std(lk_mf_list)/np.sqrt(num_repetitions)))
print("Mean GF likelihood: {} += {}".format(np.mean(lk_gf_list), np.std(lk_gf_list)/np.sqrt(num_repetitions)))