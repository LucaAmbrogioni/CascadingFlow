import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from modules.models import DynamicModel
from modules.models import MeanField, MultivariateNormal, GlobalFlow, Autoregressive
from modules.training_tools import variational_update
from modules.eval_utils import evaluate_model
from modules.plot_tools import plot_model
from modules.networks import TriResNet, ASVIupdate, LinearNet, DeepNet

def rum_timeseries_experiment(exp_name, num_repetitions, num_iterations, batch_size, transition_model,
                              dist, emission_model, emission_dist, d_x,
                              sigma, initial_sigma, observation_gain, T, T_data, lk_sigma, initial_mean):

    prior_model = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                               transition=transition_model,
                               emission=emission_model,
                               emission_distribution=emission_dist,
                               observation_gain=observation_gain, T=T, initial_mean=initial_mean)

    uni_eval_cfr = []
    multi_eval_cfr = []
    pred_eval_cfr = []
    uni_eval_cfn = []
    multi_eval_cfn = []
    pred_eval_cfn = []
    uni_eval_mf = []
    multi_eval_mf = []
    pred_eval_mf = []
    uni_eval_mn = []
    multi_eval_mn = []
    pred_eval_mn = []
    uni_eval_asvi = []
    multi_eval_asvi = []
    pred_eval_asvi = []
    uni_eval_gfr = []
    multi_eval_gfr = []
    pred_eval_gfr = []
    uni_eval_gfn = []
    multi_eval_gfn = []
    pred_eval_gfn = []
    uni_eval_ar1 = []
    multi_eval_ar1 = []
    pred_eval_ar1 = []
    uni_eval_nn1 = []
    multi_eval_nn1 = []
    pred_eval_nn1 = []

    for rep in range(num_repetitions):

        print("Repetition: {}".format(rep))

        # generate ground truth
        X_true, Y, mu = prior_model.sample_observations(1)
        x = X_true[0, 0, :].detach().numpy()
        y = Y[0, :].detach().numpy()
        data = Y[0, :T_data].view((1, T_data))
        out_data = Y[0, T_data:].view((1, Y.shape[1] - T_data))

        #plt.plot(X_true.detach().numpy()[0,0,:])
        #plt.plot(X_true.detach().numpy()[0, 1, :])
        #plt.show()

        ### Cascading flow ###
        print("Train cascading flows")
        transformations = [TriResNet(d_x=d_x, d_epsilon=10, epsilon_nu=0.1, in_pre_lambda=4.) for _ in range(T)]
        variational_model = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                                         transition=transition_model,
                                         emission=emission_model, emission_distribution=emission_dist,
                                         observation_gain=observation_gain, T=T,
                                         transformations=transformations, initial_mean=initial_mean)

        loss_list = []
        params_list = [list(tr.parameters()) for tr in transformations]
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
        uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
                                                emission_model=emission_model,
                                                emission_distribution=emission_dist,
                                                scale=lk_sigma, out_data=out_data, T_data=T_data)
        uni_eval_cfr.append(uni_lk)
        multi_eval_cfr.append(multi_lk)
        pred_eval_cfr.append(pred)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/CFr_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

        plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/CFr_rep:{}".format(exp_name, rep))

        ### Cascading flow (No residuals) ###
        print("Train cascading flows (No residuals)")
        transformations = [TriResNet(d_x=d_x, d_epsilon=10, epsilon_nu=0.1, in_pre_lambda=None) for _ in range(T)]
        variational_model = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                                         transition=transition_model,
                                         emission=emission_model,
                                         emission_distribution=emission_dist,
                                         observation_gain=observation_gain, T=T,
                                         transformations=transformations, initial_mean=initial_mean)
        loss_list = []
        params_list = [list(tr.parameters()) for tr in transformations]
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
        uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
                                                emission_model=emission_model,
                                                emission_distribution=emission_dist,
                                                scale=lk_sigma, out_data=out_data, T_data=T_data)
        uni_eval_cfn.append(uni_lk)
        multi_eval_cfn.append(multi_lk)
        pred_eval_cfn.append(pred)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/CFn_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

        plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/CFn_rep:{}".format(exp_name, rep))

        ### Mean field ###
        print("Train mean field")
        variational_model = MeanField(T=T, d_x=d_x)

        loss_list = []
        params_list = [variational_model.parameters()]
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
        uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
                                                emission_model=emission_model,
                                                emission_distribution=emission_dist,
                                                scale=lk_sigma, out_data=out_data, T_data=T_data)
        uni_eval_mf.append(uni_lk)
        multi_eval_mf.append(multi_lk)
        pred_eval_mf.append(pred)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/MF_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

        plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/MF_rep:{}".format(exp_name, rep))

        ### Multivariate normal ###
        print("Train multivariate normal")
        variational_model = MultivariateNormal(T=T, d_x=d_x)
        loss_list = []
        params_list = [variational_model.parameters()]
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
        uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
                                                emission_model=emission_model,
                                                emission_distribution=emission_dist,
                                                scale=lk_sigma, out_data=out_data, T_data=T_data)
        uni_eval_mn.append(uni_lk)
        multi_eval_mn.append(multi_lk)
        pred_eval_mn.append(pred)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/MN_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

        plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/MN_rep:{}".format(exp_name, rep))

        ### ASVI ###
        print("Train ASVI")
        mu_transformations = [ASVIupdate(l_init=3.) for _ in range(T)]
        variational_model = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
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
        uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
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

        plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/ASVI_rep:{}".format(exp_name, rep))

        ### Global flow (residual) ###
        print("Train global flow model (Residual)")
        variational_model = GlobalFlow(T=T, d_x=d_x, d_eps=10, residual=True)
        loss_list = []
        params = variational_model.transformation.parameters()
        optimizer = optim.Adam(params, lr=0.001)

        for itr in tqdm(range(num_iterations)):
            # Variational update
            loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)

            # Loss
            loss_list.append(float(loss.detach().numpy()))

        # Performance metrics
        uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
                                                emission_model=emission_model,
                                                emission_distribution=emission_dist,
                                                scale=lk_sigma, out_data=out_data, T_data=T_data)
        uni_eval_gfr.append(uni_lk)
        multi_eval_gfr.append(multi_lk)
        pred_eval_gfr.append(pred)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/GFr_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

        plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/GFr_rep:{}".format(exp_name, rep))

        ### Global flow (non-residual) ###
        print("Train global flow model (Non-residual)")
        variational_model = GlobalFlow(T=T, d_x=d_x, d_eps=10)
        loss_list = []
        params = variational_model.transformation.parameters()
        optimizer = optim.Adam(params, lr=0.001)

        for itr in tqdm(range(num_iterations)):
            # Variational update
            loss = variational_update(prior_model, variational_model, data, optimizer, batch_size)

            # Loss
            loss_list.append(float(loss.detach().numpy()))

        # Performance metrics
        uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
                                                emission_model=emission_model,
                                                emission_distribution=emission_dist,
                                                scale=lk_sigma, out_data=out_data, T_data=T_data)
        uni_eval_gfn.append(uni_lk)
        multi_eval_gfn.append(multi_lk)
        pred_eval_gfn.append(pred)

        # Plots
        plt.plot(loss_list)
        plt.savefig('{}_figures/GFn_loss_rep:{}.png'.format(exp_name, rep))
        plt.clf()

        plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/GFn_rep:{}".format(exp_name, rep))

        # ### AR(1) ###
        # print("Train AR(1)")
        # transition_models = [LinearNet(d_x=d_x) for _ in range(T)]
        # variational_model = Autoregressive(T, d_x, transition_models)
        # loss_list = []
        # params_list = [list(variational_model.parameters())] + [list(tr.parameters()) for tr in transition_models]
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
        # uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
        #                                         emission_model=emission_model,
        #                                         emission_distribution=emission_dist,
        #                                         scale=lk_sigma, out_data=out_data, T_data=T_data)
        # uni_eval_ar1.append(uni_lk)
        # multi_eval_ar1.append(multi_lk)
        # pred_eval_ar1.append(pred)
        #
        # # Plots
        # plt.plot(loss_list)
        # plt.savefig('{}_figures/AR_loss_rep:{}.png'.format(exp_name, rep))
        # plt.clf()
        #
        # plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/AR_rep:{}".format(exp_name, rep))
        #
        # ### NN(1) ###
        # print("Train NN(1)")
        # transition_models = [DeepNet(d_x=d_x, d_h=13) for _ in range(T)]
        # variational_model = Autoregressive(T, d_x, transition_models)
        # loss_list = []
        # params_list = [list(variational_model.parameters())] + [list(tr.parameters()) for tr in transition_models]
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
        # uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=1000,
        #                                         emission_model=emission_model,
        #                                         emission_distribution=emission_dist,
        #                                         scale=lk_sigma, out_data=out_data, T_data=T_data)
        # uni_eval_nn1.append(uni_lk)
        # multi_eval_nn1.append(multi_lk)
        # pred_eval_nn1.append(pred)
        #
        # # Plots
        # plt.plot(loss_list)
        # plt.savefig('{}_figures/NN_loss_rep:{}.png'.format(exp_name, rep))
        # plt.clf()
        #
        # plot_model(variational_model, X_true, K=d_x, M=100, savename="{}_figures/NN_rep:{}".format(exp_name, rep))
    uni_results = {"CFr": uni_eval_cfr, "CFn": uni_eval_cfn, "MF": uni_eval_mf, "GFr": uni_eval_gfr,
                   "GFn": uni_eval_gfn,
                   "ASVI": uni_eval_asvi, "AR1": uni_eval_ar1, "NN1": uni_eval_nn1, "MN": uni_eval_mn}
    multi_results = {"CFr": multi_eval_cfr, "CFn": multi_eval_cfn, "MF": multi_eval_mf, "GFr": multi_eval_gfr,
                     "GFn": multi_eval_gfn,
                     "ASVI": multi_eval_asvi, "AR1": multi_eval_ar1, "NN1": multi_eval_nn1, "MN": multi_eval_mn}
    pred_results = {"CFr": pred_eval_cfr, "CFn": pred_eval_cfn, "MF": pred_eval_mf, "GFr": pred_eval_gfr,
                    "GFn": pred_eval_gfn,
                    "ASVI": pred_eval_asvi, "AR1": pred_eval_ar1, "NN1": pred_eval_nn1, "MN": pred_eval_mn}

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