import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from modules.distributions import NormalDistribution
from modules.dynamics import ConvTransition
from modules.emissions import LinearEmission
from modules.eval_utils import evaluate_img_model
from modules.models import DynamicImgModel
from modules.networks import ASVIupdate
from modules.plot_tools import plot_model
from modules.training_tools import variational_update


def rum_timeseries_img_experiment(exp_name, num_repetitions, num_iterations, batch_size, transition_model,
                                  dist, emission_model, emission_dist, d_x,
                                  sigma, initial_sigma, observation_gain, T, T_data, lk_sigma, initial_mean):
    prior_model = DynamicImgModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                                  transition=transition_model,
                                  emission=emission_model,
                                  emission_distribution=emission_dist,
                                  observation_gain=observation_gain, T=T, initial_mean=initial_mean)

    uni_eval_asvi = []
    multi_eval_asvi = []
    pred_eval_asvi = []

    for rep in range(num_repetitions):

        print("Repetition: {}".format(rep))

        # generate ground truth
        with torch.no_grad():
            X_true, Y, mu = prior_model.sample_observations(1)
            data = Y[0, :T_data].view((1, T_data, -1))
            out_data = Y[0, T_data:].view((1, Y.shape[1] - T_data, -1))

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
            loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)

            # Loss
            loss_list.append(float(loss.detach().numpy()))

        # Performance metrics
        uni_lk, multi_lk, pred = evaluate_img_model(variational_model, X_true, M=5000,
                                                emission_model=emission_model,
                                                emission_distribution=emission_dist,
                                                scale=lk_sigma, out_data=out_data, T_data=T_data)
        uni_eval_asvi.append(uni_lk)
        multi_eval_asvi.append(multi_lk)
        pred_eval_asvi.append(pred)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/ASVI_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

    uni_results = {"ASVI": uni_eval_asvi}
    multi_results = {"ASVI": multi_eval_asvi}
    pred_results = {"ASVI": pred_eval_asvi}

    import pickle
    pickle_out = open("{}_results/uni_results.pickle".format(exp_name), "wb")
    pickle.dump(uni_results, pickle_out)
    pickle_out.close()

    pickle_out = open("{}_results/multi_results.pickle".format(exp_name), "wb")
    pickle.dump(multi_results, pickle_out)
    pickle_out.close()

    pickle_out = open("{}_results/pred_results.pickle".format(exp_name), "wb")
    pickle.dump(pred_results, pickle_out)
    pickle_out.close()


# Defining dynamical and emission model
model_name = "conv"
lik_name = "r"
exp_name = model_name + "_" + lik_name

if model_name == "conv":
    T = 40  # 60
    T_data = 20
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
    transition_model = ConvTransition(in_ch, out_ch, kernel_size, pad=padding, dt=dt)

if lik_name == "r":
    observation_gain = 1.
    emission_model = LinearEmission(28 * 28, out_features=10)
    emission_dist = NormalDistribution(scale=lk_sigma)

num_repetitions = 10
num_iterations = 200  # 8000
batch_size = 50

rum_timeseries_img_experiment(exp_name, num_repetitions, num_iterations, batch_size, transition_model,
                              dist, emission_model, emission_dist, d_x,
                              sigma, initial_sigma, observation_gain, T, T_data, lk_sigma, initial_mean)
