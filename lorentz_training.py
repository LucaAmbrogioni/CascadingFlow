import os

import numpy as np

from modules.distributions import NormalDistribution, BernoulliDistribution
from modules.dynamics import LorentzTransition, RNNTransition, BrusselatorTransition, VolterraTransition
from modules.emissions import SingleCoordinateEmission
from modules.experiments import rum_timeseries_experiment

# Defining dynamical and emission model
model_name = "bruss"
lik_name = "c"
exp_name = model_name + "_" + lik_name

if not os.path.isdir(f'{exp_name}_figures'):
    os.makedirs(f'{exp_name}_figures')

if not os.path.isdir(f'{exp_name}_results'):
    os.makedirs(f'{exp_name}_results')

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
elif model_name == "vol":
    T = 100
    T_data = 50
    dt = 1.
    sigma = np.sqrt(dt)*0.5
    initial_sigma = 3.
    initial_mean = 20.
    d_x = 2 #Number of latent variables
    dist = NormalDistribution()
    lk_sigma = 3.
    transition_model = VolterraTransition(dt=dt)
elif model_name == "bruss":
    T = 80
    T_data = 40
    dt = 0.1
    sigma = np.sqrt(dt)*0.1
    initial_sigma = 1.
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

num_repetitions = 10
num_iterations = 5000 #8000
batch_size = 50

rum_timeseries_experiment(exp_name, num_repetitions, num_iterations, batch_size, transition_model,
                          dist, emission_model, emission_dist, d_x,
                          sigma, initial_sigma, observation_gain, T, T_data, lk_sigma, initial_mean)
    
