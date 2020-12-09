import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules.distributions import NormalDistribution, BernoulliDistribution
from modules.models import DynamicModel
from modules.dynamics import LorentzTransition
from modules.emissions import SingleCoordinateEmission
from modules.models import MeanField

# Defining dynamical and emission model
T = 40
dt = 0.02
sigma = np.sqrt(dt)*2.
initial_sigma = 1.
observation_gain = 2.
d_x = 3 #Number of latent variables
dist = NormalDistribution()
bm = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                  transition=LorentzTransition(dt=dt),
                  emission=SingleCoordinateEmission(k=0, gain=observation_gain),
                  emission_distribution=BernoulliDistribution(), observation_gain=observation_gain, T=T)

num_repetitions = 10
num_iterations = 10000
batch_size = 50
prior_model = bm

for rep in range(num_repetitions):

    print("Repetition: {}".format(rep))

    # generate ground truth
    X_true, Y, mu =  bm.sample_observations(1)
    x = X_true[0, 0, :].detach().numpy()
    y = Y[0, :].detach().numpy()
    data = Y[0, :].view((1, T))


    ### Mean field ###
    variational_model = MeanField(T=T, d_x=d_x)
    loss_list_mf = []
    parames_list = [variational_model.parameters()]
    params = []
    for p in parames_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    print("Train mean field model")
    for itr in tqdm(range(num_iterations)):
        # Gradient reset
        optimizer.zero_grad()

        # Variational loss
        X, mu, x_pre, log_jacobian, epsilon_loss = variational_model.sample_timeseries(batch_size)
        log_q = variational_model.evaluate_avg_joint_log_prob(X, None, mu, x_pre=x_pre, log_jacobian=log_jacobian,
                                                              epsilon_loss=epsilon_loss)
        log_p = prior_model.evaluate_avg_joint_log_prob(X, data, mu)
        loss = (log_q - log_p)

        # Update
        loss.backward()
        optimizer.step()
        loss_list_mf.append(float(loss.detach().numpy()))

    
