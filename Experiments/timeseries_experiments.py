import numpy as np

from modules.distributions import NormalDistribution, BernoulliDistribution
from modules.dynamics import LorentzTransition, RNNTransition, BrusselatorTransition, VolterraTransition, VanDerPollTransition
from modules.emissions import SingleCoordinateEmission
from modules.experiments import rum_timeseries_experiment

# Defining dynamical and emission model
model_name = "lz"
lik_name = "r"
exp_name = model_name + "_" + lik_name

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
    initial_mean = 20.
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

num_repetitions = 10
num_iterations = 2 #8000
batch_size = 50

rum_timeseries_experiment(exp_name, num_repetitions, num_iterations, batch_size, transition_model,
                          dist, emission_model, emission_dist, d_x,
                          sigma, initial_sigma, observation_gain, T, T_data, lk_sigma, initial_mean)
    
