from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from modules.networks import LinearGaussianTree, TriResNet, ASVIupdate
from modules.models import ColliderModel, MeanField, GlobalFlow, MultivariateNormal
from modules.distributions import NormalDistribution
from modules.eval_utils import evaluate_multi_likelihood

# Parameters
depth = 2 #3
#join_link = lambda x, y: x - y
join_link = lambda x, y, k=2.: torch.tanh(k*x) - torch.tanh(k*y)
dist = NormalDistribution()
num_iterations = 7000 #10000
batch_size = 80
sigma = 0.05
in_sigma= 0.1 #0.2
num_samples = 20000

# Prior model
prior_model = ColliderModel(depth=depth, sigma=sigma, in_sigma=in_sigma, join_link=join_link,
                            transition_distribution=NormalDistribution())

# Data
true_smpl,_,_,_ = prior_model.sample(1)
pr_smpl,_,_,_ = prior_model.sample(num_samples)
value = join_link(true_smpl[-1][0,-2], true_smpl[-1][0,-1]).detach().numpy() + np.random.normal(0,sigma)
print(value)
y = torch.tensor(np.array([value])).type(torch.float32)

num_repetitions = 15
print("Depth: {}".format(depth))

lk_list = []
lk_asvi_list = []
lk_mf_list = []
lk_gf_list = []
lk_mn_list = []

for _ in range(num_repetitions):
    ### Cascading flows ###
    d_eps = 10
    tree = LinearGaussianTree(node_size=d_eps,depth=depth,in_scale=0.3,scale=0.5, in_w = 4.) #3
    transformations = [TriResNet(d_x=1, d_epsilon=d_eps, epsilon_nu=0., in_pre_lambda=3., scale_w=0.8,) for _ in range(2**depth-1)] #0.8
    post_model = ColliderModel(depth=depth, sigma=sigma, in_sigma=in_sigma, join_link=join_link,
                               transition_distribution=dist,
                               transformations=transformations, eps_generator=tree)
    loss_list1 = []
    parames_list = [tr.parameters() for tr in transformations] #+ [tree.parameters()]
    params = []
    for p in parames_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    print("Train Cascading Flow model")
    for itr in tqdm(range(num_iterations)):
        # Gradient reset
        optimizer.zero_grad()

        # Variational loss
        samples, samples_pre, log_jacobian, epsilon_loss = post_model.sample(batch_size)
        log_q = post_model.evaluate_avg_joint_log_prob(samples, None, samples_pre, log_jacobian=log_jacobian
                                                       ,epsilon_loss=epsilon_loss)
        log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
        loss = (log_q - log_p)

        # Update
        loss.backward()
        optimizer.step()
        loss_list1.append(float(loss.detach().numpy()))
        #if itr % 100 == 0:
        #    print(tree.weights)

    ### ASVI ###
    mu_transformations = [ASVIupdate(l_init=3.) for _ in range(2**depth-1)]
    post_model_asvi = ColliderModel(depth=depth, sigma=sigma, in_sigma=in_sigma, join_link=join_link,
                               transition_distribution=dist,
                               mu_transformations=mu_transformations)
    loss_list2 = []
    parames_list = [tr.parameters() for tr in mu_transformations]
    params = []
    for p in parames_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    print("Train Cascading Flow model (no coupling)")
    for itr in tqdm(range(num_iterations)):
        # Gradient reset
        optimizer.zero_grad()

        # Variational loss
        samples, samples_pre, log_jacobian, epsilon_loss = post_model_asvi.sample(batch_size)
        log_q = post_model_asvi.evaluate_avg_joint_log_prob(samples, None, samples_pre, log_jacobian=log_jacobian,
                                                           epsilon_loss=epsilon_loss)
        log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
        loss = (log_q - log_p)

        # Update
        loss.backward()
        optimizer.step()
        loss_list2.append(float(loss.detach().numpy()))
    #
    ### Mean field ###
    post_model_mf = MeanField(T=2**depth-2, d_x=1)
    loss_list3 = []
    parames_list = [post_model_mf.parameters()]
    params = []
    for p in parames_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    print("Train Mean Field model")
    for itr in tqdm(range(num_iterations)):
        # Gradient reset
        optimizer.zero_grad()

        # Variational loss
        x, _, _, _, _ = post_model_mf.sample_timeseries(batch_size)
        samples = post_model_mf.reshape_collider_samples(x, depth)
        log_q = post_model_mf.evaluate_avg_joint_log_prob(x, None, 0.)
        log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
        loss = (log_q - log_p)

        # Update
        loss.backward()
        optimizer.step()
        loss_list3.append(float(loss.detach().numpy()))
    #
    ### Global flow ###
    post_model_gf = GlobalFlow(T=2**depth-2, d_x=1, d_eps=5)
    loss_list4 = []
    parames_list = [post_model_gf.parameters()]
    params = []
    for p in parames_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    print("Train Global flow")
    for itr in tqdm(range(num_iterations)):
        # Gradient reset
        optimizer.zero_grad()

        # Variational loss
        x, _, samples_pre, log_jacobian, epsilon_loss = post_model_gf.sample_timeseries(batch_size)
        samples = post_model_gf.reshape_collider_samples(x, depth)
        log_q = post_model_gf.evaluate_avg_joint_log_prob(x, None, 0., samples_pre, log_jacobian=log_jacobian)
    #                                                       , epsilon_loss=epsilon_loss)
        log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
        loss = (log_q - log_p)

        # Update
        loss.backward()
        optimizer.step()
        loss_list4.append(float(loss.detach().numpy()))

    ### Multivariate Normal ###
    post_model_mn = MultivariateNormal(T=2**depth-2, d_x=1)
    loss_list5 = []
    parames_list = [post_model_mn.parameters()]
    params = []
    for p in parames_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    print("Train Mean Field model")
    for itr in tqdm(range(num_iterations)):
        # Gradient reset
        optimizer.zero_grad()

        # Variational loss
        x, _, _, _, _ = post_model_mn.sample_timeseries(batch_size)
        samples = post_model_mn.reshape_collider_samples(x, depth)
        log_q = post_model_mn.evaluate_avg_joint_log_prob(x, None, 0.)
        log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
        loss = (log_q - log_p)

        # Update
        loss.backward()
        optimizer.step()
        loss_list5.append(float(loss.detach().numpy()))

    # Performance metrics
    #evaluate_likelihood(X, x_true)
    #uni_lk, multi_lk, pred = evaluate_model(variational_model, X_true, M=5000,
    #                                        emission_model=emission_model,
    #                                        emission_distribution=emission_dist,
    #                                        scale=lk_sigma, out_data=out_data, T_data=T_data)

    #plt.plot(loss_list1)
    #plt.plot(loss_list2)
    #plt.plot(loss_list3)
    #plt.plot(loss_list4)
    #plt.show()

    corr_list = []
    N_itr = 10
    # CF
    smpl,_,_,_ = post_model.sample(num_samples)
    smpl = torch.cat(smpl,1).detach().numpy()
    # ASVI
    smpl_asvi, _, _, _ = post_model_asvi.sample(num_samples)
    smpl_asvi = torch.cat(smpl_asvi, 1).detach().numpy()
    # MF
    smpl_mf,_,_,_,_ = post_model_mf.sample_timeseries(num_samples)
    smpl_mf = smpl_mf.squeeze().detach().numpy()
    #GF
    smpl_gf,_,_,_,_ = post_model_gf.sample_timeseries(num_samples)
    smpl_gf = smpl_gf.squeeze().detach().numpy()
    #MN
    smpl_mn,_,_,_,_ = post_model_mn.sample_timeseries(num_samples)
    smpl_mn = smpl_mn.squeeze().detach().numpy()

    re_true_smpl = torch.cat(true_smpl,1).detach().numpy()

    lk = evaluate_multi_likelihood(smpl, re_true_smpl)
    lk_asvi = evaluate_multi_likelihood(smpl_asvi, re_true_smpl)
    lk_mf = evaluate_multi_likelihood(smpl_mf, re_true_smpl)
    lk_gf = evaluate_multi_likelihood(smpl_gf, re_true_smpl)
    lk_mn = evaluate_multi_likelihood(smpl_mn, re_true_smpl)
    print("CF likelihood: {}".format(lk))
    print("ASVI likelihood: {}".format(lk_asvi))
    print("MF likelihood: {}".format(lk_mf))
    print("GF likelihood: {}".format(lk_gf))
    print("MN likelihood: {}".format(lk_mn))

    lk_list.append(lk)
    lk_asvi_list.append(lk_asvi)
    lk_mf_list.append(lk_mf)
    lk_gf_list.append(lk_gf)
    lk_mn_list.append(lk_mn)

# corr1 = [np.corrcoef(smpl[:,-1], smpl[:,k])[0,1] for k in range(smpl.shape[1])]
# #corr2 = [np.corrcoef(smpl_cfn[:,-1], smpl_cfn[:,k])[0,1] for k in range(smpl.shape[1])]
# p_smpl = torch.cat(pr_smpl,1)
# pr_corr = [np.corrcoef(p_smpl[:,-1], p_smpl[:,k])[0,1] for k in range(smpl.shape[1])]
# plt.plot(corr1, c="r")
# #plt.plot(corr2, c="m")
# plt.plot(pr_corr, c="k", ls="--")
# plt.axhline(y=0., color='k', linestyle='--', lw=2)
# plt.show()
#
# ## True posterior ##
# density = lambda x,y,s=in_sigma: np.exp(-(x**2+y**2)/(2*s**2))/np.sqrt(2*np.pi*s**2)
# mu_link = lambda x,y: join_link(x,y)
# s_link = lambda x,y: sigma
# lk = lambda x,y,z: np.exp(-(z - mu_link(x,y))**2/(2*s_link(x,y)**2))/np.sqrt(2*np.pi*s_link(x,y)**2)
# post = lambda x,y,z: density(x,y)*lk(x,y,z)
#
# d = 4.
# M = 300
# x_range = np.linspace(-d,d,M)
# y_range = np.linspace(-d,d,M)
#
# mesh1, mesh2 = np.meshgrid(x_range, y_range)
#
# data = value
# posterior = density(mesh1, mesh2)*lk(mesh1,mesh2,data)
# posterior = posterior/np.sum(posterior)
#
# plt.imshow(posterior, extent=[-d,d,-d,d], origin="lower", cmap="Greys")
# plt.scatter((smpl[:,-2]), (smpl[:,-1]), c="r", alpha=0.002)
# plt.scatter((true_smpl[-1][:,-2]), (true_smpl[-1][:,-1]), c="k")
# plt.xlim(-d,d)
# plt.ylim(-d,d)
# plt.show()
#
# plt.imshow(posterior, extent=[-d,d,-d,d], origin="lower", cmap="Greys")
# plt.scatter((smpl_mf[:,-2]), (smpl_mf[:,-1]), c="b", alpha=0.002)
# plt.scatter((true_smpl[-1][:,-2]), (true_smpl[-1][:,-1]), c="k")
# plt.xlim(-d,d)
# plt.ylim(-d,d)
# plt.show()
#
# plt.imshow(posterior, extent=[-d,d,-d,d], origin="lower", cmap="Greys")
# plt.scatter((smpl_mn[:,-2]), (smpl_mn[:,-1]), c="g", alpha=0.002)
# plt.scatter((true_smpl[-1][:,-2]), (true_smpl[-1][:,-1]), c="k")
# plt.xlim(-d,d)
# plt.ylim(-d,d)
# plt.show()
#
# plt.imshow(posterior, extent=[-d,d,-d,d], origin="lower", cmap="Greys")
# plt.scatter((smpl_gf[:,-2]), (smpl_gf[:,-1]), c="c", alpha=0.002)
# plt.scatter((true_smpl[-1][:,-2]), (true_smpl[-1][:,-1]), c="k")
# plt.xlim(-d,d)
# plt.ylim(-d,d)
# plt.show()
#
# # plt.scatter((pr_smpl[-1][:,-1]), (pr_smpl[-1][:,-2]), c="b", alpha=0.01)
# # plt.scatter((smpl_cfn[:,-1]), (smpl_cfn[:,-2]), c="m", alpha=0.01)
# # plt.scatter((true_smpl[-1][:,-1]), (true_smpl[-1][:,-2]), c="k")
# # plt.show()
# #
# # plt.scatter((pr_smpl[-1][:,-1]), (pr_smpl[-1][:,-2]), c="b", alpha=0.01)
# # plt.scatter((smpl_mf[:,-1]), (smpl_mf[:,-2]), c="g", alpha=0.01)
# # plt.scatter((true_smpl[-1][:,-1]), (true_smpl[-1][:,-2]), c="k")
# # plt.show()
# #
# #plt.scatter((pr_smpl[-1][:,-1]), (pr_smpl[-1][:,-2]), c="b", alpha=0.01)
# #plt.scatter((smpl_gf[:,-1]), (smpl[:,-2]), c="c", alpha=0.01)
# #plt.scatter((true_smpl[-1][:,-1]), (true_smpl[-1][:,-2]), c="k")
# #plt.show()
# #
# # #plt.hist(join_link(pr_smpl[-1][:,-1],pr_smpl[-1][:,-2]),30, c="b")
# plt.hist(join_link(smpl[:,-2],smpl[:,-1]),30, alpha=0.5, color="r")
# # plt.hist(join_link(smpl_cfn[:,-1],smpl_cfn[:,-2]),30, alpha=0.5, color="m")
# # plt.hist(join_link(smpl_mf[:,-1],smpl_mf[:,-2]),30, alpha=0.5, color="g")
# #plt.hist(join_link(smpl_gf[:,-1],smpl_gf[:,-2]),30, alpha=0.5, color="c")
# plt.axvline(x=value, color='k', linestyle='--', lw=2)
# plt.show()

print("Mean CF likelihood: {} += {}".format(np.mean(lk_list), np.std(lk_list)/np.sqrt(num_repetitions)))
print("Mean ASVI likelihood: {} += {}".format(np.mean(lk_asvi_list), np.std(lk_asvi_list)/np.sqrt(num_repetitions)))
print("Mean MF likelihood: {} += {}".format(np.mean(lk_mf_list), np.std(lk_mf_list)/np.sqrt(num_repetitions)))
print("Mean GF likelihood: {} += {}".format(np.mean(lk_gf_list), np.std(lk_gf_list)/np.sqrt(num_repetitions)))
print("Mean MN likelihood: {} += {}".format(np.mean(lk_mn_list), np.std(lk_mn_list)/np.sqrt(num_repetitions)))