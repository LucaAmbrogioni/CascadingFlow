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
from modules.models import MeanField, MultivariateNormal, GlobalFlow, Autoregressive
from modules.training_tools import variational_update
from modules.eval_utils import evaluate_model
from modules.plot_tools import plot_model
from modules.networks import TriResNet, ASVIupdate, LinearNet, DeepNet

# Defining dynamical and emission model
T = 40
dt = 0.02
sigma = np.sqrt(dt)*2.
initial_sigma = 1.
observation_gain = 1.
d_x = 1 #Number of latent variables
dist = NormalDistribution()
transition_model = lambda x,m: x
bm = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                  transition=transition_model,
                  emission=SingleCoordinateEmission(k=0, gain=observation_gain),
                  emission_distribution=NormalDistribution(0.1), observation_gain=observation_gain, T=T)

num_repetitions = 10
num_iterations = 2000 #10000
batch_size = 50
prior_model = bm



for rep in range(num_repetitions):

    print("Repetition: {}".format(rep))

    # generate ground truth
    X_true, Y, mu = bm.sample_observations(1)
    x = X_true[0, 0, :].detach().numpy()
    y = Y[0, :].detach().numpy()
    data = Y[0, :].view((1, T))

    x = X_true[0, 0, :].detach().numpy()
    y = Y[0, :].detach().numpy()
    plt.scatter(range(T), y)
    plt.plot(x)
    plt.show()

    # ### Multivariate normal ###
    # print("Train multivariate normal")
    # variational_model = MultivariateNormal(T=T, d_x=d_x)
    # loss_list = []
    # uni_eval_cf = []
    # multi_eval_cf = []
    # params_list = [variational_model.parameters()]
    # params = []
    # for p in params_list:
    #     params += p
    # optimizer = optim.Adam(params, lr=0.001)
    #
    # for itr in tqdm(range(num_iterations)):
    #     # Variational update
    #     loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)
    #
    #     # Loss
    #     loss_list.append(float(loss.detach().numpy()))
    #
    # # Performance metrics
    # uni_lk, multi_lk = evaluate_model(variational_model, X_true, M=1000)
    # uni_eval_cf.append(uni_lk)
    # multi_eval_cf.append(multi_lk)
    #
    # # Plots
    # plt.plot(loss_list)
    # plt.show()
    #
    # plot_model(variational_model, X_true, K=d_x, M=100)
    #
    # ### Global flow (residual) ###
    # print("Train global flow model (Residual)")
    # variational_model = GlobalFlow(T=T, d_x=d_x, d_eps=10, residual=True)
    # loss_list = []
    # uni_eval_cf = []
    # multi_eval_cf = []
    # params = variational_model.transformation.parameters()
    # optimizer = optim.Adam(params, lr=0.001)
    #
    # for itr in tqdm(range(num_iterations)):
    #     # Variational update
    #     loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)
    #
    #     # Loss
    #     loss_list.append(float(loss.detach().numpy()))
    #
    # # Performance metrics
    # uni_lk, multi_lk = evaluate_model(variational_model, X_true, M=1000)
    # uni_eval_cf.append(uni_lk)
    # multi_eval_cf.append(multi_lk)
    #
    # # Plots
    # plt.plot(loss_list)
    # plt.show()
    #
    # plot_model(variational_model, X_true, K=d_x, M=100)
    #
    ### AR(1) ###
    print("Train AR(1)")
    transition_models = [LinearNet(d_x=d_x) for _ in range(T)]
    variational_model = Autoregressive(T, d_x, transition_models)
    loss_list = []
    uni_eval_cf = []
    multi_eval_cf = []
    params_list = [list(variational_model.parameters())] + [list(tr.parameters()) for tr in transition_models]
    params = []
    for p in params_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    for itr in tqdm(range(num_iterations)):
        # Variational update
        loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)

        # Loss
        loss_list.append(float(loss.detach().numpy()))

    # Performance metrics
    uni_lk, multi_lk = evaluate_model(variational_model, X_true, M=1000)
    uni_eval_cf.append(uni_lk)
    multi_eval_cf.append(multi_lk)

    # Plots
    plt.plot(loss_list)
    plt.show()

    plot_model(variational_model, X_true, K=d_x, M=100)

    ### NN(1) ###
    print("Train NN(1)")
    transition_models = [DeepNet(d_x=d_x, d_h=20) for _ in range(T)]
    variational_model = Autoregressive(T, d_x, transition_models)
    loss_list = []
    uni_eval_cf = []
    multi_eval_cf = []
    params_list = [list(variational_model.parameters())] + [list(tr.parameters()) for tr in transition_models]
    params = []
    for p in params_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    for itr in tqdm(range(num_iterations)):
        # Variational update
        loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)

        # Loss
        loss_list.append(float(loss.detach().numpy()))

    # Performance metrics
    uni_lk, multi_lk = evaluate_model(variational_model, X_true, M=1000)
    uni_eval_cf.append(uni_lk)
    multi_eval_cf.append(multi_lk)

    # Plots
    plt.plot(loss_list)
    plt.show()

    plot_model(variational_model, X_true, K=d_x, M=100)

    
