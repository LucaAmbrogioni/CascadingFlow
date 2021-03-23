import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import os

from modules.distributions import NormalDistribution
from modules.dynamics import ConvTransition
from modules.emissions import LinearEmission
from modules.eval_utils import evaluate_img_model
from modules.models import DynamicImgModel
from modules.networks import ASVIupdate
from modules.plot_tools import plot_model
from modules.training_tools import variational_img_update
from modules.networks import TriResNet

def rum_timeseries_img_experiment(exp_name, num_repetitions, num_iterations, batch_size, transition_model,
                                  dist, emission_model, emission_dist, d_x,
                                  sigma, initial_sigma, observation_gain, T, bin_list, lk_sigma, initial_mean):
    prior_model = DynamicImgModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                                  transition=transition_model,
                                  emission=emission_model,
                                  emission_distribution=emission_dist,
                                  observation_gain=observation_gain, T=T, initial_mean=initial_mean, is_prior=True)

    if not os.path.isdir(f'{exp_name}_figures'):
        os.makedirs(f'{exp_name}_figures')

    if not os.path.isdir(f'{exp_name}_results'):
        os.makedirs(f'{exp_name}_results')

    uni_eval_asvi = []

    for rep in range(num_repetitions):

        print("Repetition: {}".format(rep))

        # generate ground truth
        with torch.no_grad():
            X_true, Y, mu = prior_model.sample_observations(1)
            # X_true shape: (N_samples, TimeSteps, Channels, Width, Height)
            # Y shape : (N_samples, TimeSteps, OutFeatures)

        # plt.plot(X_true.detach().numpy()[0,0,:])
        # plt.plot(X_true.detach().numpy()[0, 1, :])
        # plt.show()

        ### ASVI ###
        print("Train ASVI")
        mu_transformations = [ASVIupdate(l_init=3.) for _ in range(T)]
        variational_model = DynamicImgModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                                         transition=transition_model,
                                         emission=emission_model,
                                         emission_distribution=emission_dist,
                                         observation_gain=observation_gain, T=T,
                                         mu_transformations=mu_transformations,
                                         initial_mean=initial_mean)
        loss_list = []
        params_list = [list(tr.parameters()) for tr in mu_transformations]
        params = []
        for p in params_list:
            params += p
        optimizer = optim.Adam(params, lr=0.001)

        for itr in tqdm(range(num_iterations)):
            # Variational update
            loss = variational_img_update(prior_model, variational_model, Y, bin_list, optimizer, batch_size)

            # Loss
            loss_list.append(float(loss.detach().numpy()))

        # Performance metrics
        uni_lk = evaluate_img_model(variational_model, X_true, M=5000)

        uni_eval_asvi.append(uni_lk)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/ASVI_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

    uni_results = {"ASVI": uni_eval_asvi}

    import pickle
    pickle_out = open("{}_results/uni_results.pickle".format(exp_name), "wb")
    pickle.dump(uni_results, pickle_out)
    pickle_out.close()


# Defining dynamical and emission model
model_name = "conv"
lik_name = "r"
exp_name = model_name + "_" + lik_name

if model_name == "conv":
    T = 5  # 60
    #T_data = 4
    bin_list = [0 for _ in range(T-1)]
    bin_list.append(1)
    dt = 0.2
    sigma = np.sqrt(dt) * 0.1  # 0.5
    initial_sigma = 1.
    initial_mean = 0.
    in_ch = 1
    out_ch = 1
    kernel_size = 3
    padding = 1
    d_x = 28
    dist = NormalDistribution()
    lk_sigma = 1.
    transition_model = ConvTransition(in_ch, out_ch, kernel_size, pad=padding)
    out_features = 512 # size after linear emission

if lik_name == "r":
    observation_gain = 1.
    emission_model = LinearEmission(d_x * d_x, out_features=out_features)
    emission_dist = NormalDistribution(scale=lk_sigma)

num_repetitions = 10
num_iterations = 2  # 8000
batch_size = 50

rum_timeseries_img_experiment(exp_name, num_repetitions, num_iterations, batch_size, transition_model,
                              dist, emission_model, emission_dist, d_x,
                              sigma, initial_sigma, observation_gain, T, bin_list, lk_sigma, initial_mean)
