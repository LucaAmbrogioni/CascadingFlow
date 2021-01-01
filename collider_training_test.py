from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from modules.networks import LinearGaussianTree, TriResNet
from modules.models import ColliderModel, MeanField, GlobalFlow
from modules.distributions import NormalDistribution

# Parameters
depth = 3
join_link = lambda x, y: x+y #x + x**2 + y + y**2
dist = NormalDistribution()
num_iterations = 10000 #25000
batch_size = 80
sigma = 0.1
in_sigma= 0.6
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

### Cascading flows ###
d_eps = 10
tree = LinearGaussianTree(node_size=d_eps,depth=depth,in_scale=0.3,scale=0.5)
transformations = [TriResNet(d_x=1, d_epsilon=d_eps, epsilon_nu=0., in_pre_lambda=1., scale_w=0.8,) for _ in range(2**depth-1)]
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

# ### Cascading flows (No coupling) ###
# d_eps = 5
# transformations = [TriResNet(d_x=1, d_epsilon=d_eps, epsilon_nu=0.1, in_pre_lambda=1., scale_w=0.8,) for _ in range(2**depth-1)]
# post_model_cfn = ColliderModel(depth=depth, sigma=sigma, in_sigma=in_sigma, join_link=join_link,
#                            transition_distribution=dist,
#                            transformations=transformations)
# loss_list2 = []
# parames_list = [tr.parameters() for tr in transformations]
# params = []
# for p in parames_list:
#     params += p
# optimizer = optim.Adam(params, lr=0.001)
#
# print("Train Cascading Flow model (no coupling)")
# for itr in tqdm(range(num_iterations)):
#     # Gradient reset
#     optimizer.zero_grad()
#
#     # Variational loss
#     samples, samples_pre, log_jacobian, epsilon_loss = post_model_cfn.sample(batch_size)
#     log_q = post_model_cfn.evaluate_avg_joint_log_prob(samples, None, samples_pre, log_jacobian=log_jacobian,
#                                                        epsilon_loss=epsilon_loss)
#     log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
#     loss = (log_q - log_p)
#
#     # Update
#     loss.backward()
#     optimizer.step()
#     loss_list2.append(float(loss.detach().numpy()))
#
# ### Mean field ###
# post_model_mf = MeanField(T=2**depth-2, d_x=1)
# loss_list3 = []
# parames_list = [post_model_mf.parameters()]
# params = []
# for p in parames_list:
#     params += p
# optimizer = optim.Adam(params, lr=0.001)
#
# print("Train Mean Field model")
# for itr in tqdm(range(num_iterations)):
#     # Gradient reset
#     optimizer.zero_grad()
#
#     # Variational loss
#     x, _, _, _, _ = post_model_mf.sample_timeseries(batch_size)
#     samples = post_model_mf.reshape_collider_samples(x, depth)
#     log_q = post_model_mf.evaluate_avg_joint_log_prob(x, None, 0.)
#     log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
#     loss = (log_q - log_p)
#
#     # Update
#     loss.backward()
#     optimizer.step()
#     loss_list3.append(float(loss.detach().numpy()))
#
# ### Global flow ###
# post_model_gf = GlobalFlow(T=2**depth-2, d_x=1, d_eps=5)
# loss_list4 = []
# parames_list = [post_model_gf.parameters()]
# params = []
# for p in parames_list:
#     params += p
# optimizer = optim.Adam(params, lr=0.001)
#
# print("Train Global flow")
# for itr in tqdm(range(num_iterations)):
#     # Gradient reset
#     optimizer.zero_grad()
#
#     # Variational loss
#     x, _, samples_pre, log_jacobian, epsilon_loss = post_model_gf.sample_timeseries(batch_size)
#     samples = post_model_gf.reshape_collider_samples(x, depth)
#     log_q = post_model_gf.evaluate_avg_joint_log_prob(x, None, 0., samples_pre, log_jacobian=log_jacobian)
# #                                                       , epsilon_loss=epsilon_loss)
#     log_p = prior_model.evaluate_avg_joint_log_prob(samples, y)
#     loss = (log_q - log_p)
#
#     # Update
#     loss.backward()
#     optimizer.step()
#     loss_list4.append(float(loss.detach().numpy()))

plt.plot(loss_list1)
#plt.plot(loss_list2)
#plt.plot(loss_list3)
#plt.plot(loss_list4)
plt.show()

corr_list = []
N_itr = 10
smpl,_,_,_ = post_model.sample(num_samples)
smpl = torch.cat(smpl,1).detach().numpy()
# smpl_cfn,_,_,_ = post_model_cfn.sample(num_samples)
# smpl_cfn = torch.cat(smpl_cfn,1).detach().numpy()
# smpl_mf,_,_,_,_ = post_model_mf.sample_timeseries(num_samples)
# smpl_mf = smpl_mf.squeeze().detach().numpy()
#smpl_gf,_,_,_,_ = post_model_gf.sample_timeseries(num_samples)
#smpl_gf = smpl_gf.squeeze().detach().numpy()
corr1 = [np.corrcoef(smpl[:,-1], smpl[:,k])[0,1] for k in range(smpl.shape[1])]
#corr2 = [np.corrcoef(smpl_cfn[:,-1], smpl_cfn[:,k])[0,1] for k in range(smpl.shape[1])]
p_smpl = torch.cat(pr_smpl,1)
pr_corr = [np.corrcoef(p_smpl[:,-1], p_smpl[:,k])[0,1] for k in range(smpl.shape[1])]
plt.plot(corr1, c="r")
#plt.plot(corr2, c="m")
plt.plot(pr_corr, c="k", ls="--")
plt.axhline(y=0., color='k', linestyle='--', lw=2)
plt.show()

plt.scatter((pr_smpl[-1][:,-2]), (pr_smpl[-1][:,-1]), c="b", alpha=0.01)
plt.scatter((smpl[:,-2]), (smpl[:,-1]), c="r", alpha=0.01)
plt.scatter((true_smpl[-1][:,-2]), (true_smpl[-1][:,-1]), c="k")
plt.show()

# plt.scatter((pr_smpl[-1][:,-1]), (pr_smpl[-1][:,-2]), c="b", alpha=0.01)
# plt.scatter((smpl_cfn[:,-1]), (smpl_cfn[:,-2]), c="m", alpha=0.01)
# plt.scatter((true_smpl[-1][:,-1]), (true_smpl[-1][:,-2]), c="k")
# plt.show()
#
# plt.scatter((pr_smpl[-1][:,-1]), (pr_smpl[-1][:,-2]), c="b", alpha=0.01)
# plt.scatter((smpl_mf[:,-1]), (smpl_mf[:,-2]), c="g", alpha=0.01)
# plt.scatter((true_smpl[-1][:,-1]), (true_smpl[-1][:,-2]), c="k")
# plt.show()
#
#plt.scatter((pr_smpl[-1][:,-1]), (pr_smpl[-1][:,-2]), c="b", alpha=0.01)
#plt.scatter((smpl_gf[:,-1]), (smpl[:,-2]), c="c", alpha=0.01)
#plt.scatter((true_smpl[-1][:,-1]), (true_smpl[-1][:,-2]), c="k")
#plt.show()
#
# #plt.hist(join_link(pr_smpl[-1][:,-1],pr_smpl[-1][:,-2]),30, c="b")
plt.hist(join_link(smpl[:,-2],smpl[:,-1]),30, alpha=0.5, color="r")
# plt.hist(join_link(smpl_cfn[:,-1],smpl_cfn[:,-2]),30, alpha=0.5, color="m")
# plt.hist(join_link(smpl_mf[:,-1],smpl_mf[:,-2]),30, alpha=0.5, color="g")
#plt.hist(join_link(smpl_gf[:,-1],smpl_gf[:,-2]),30, alpha=0.5, color="c")
plt.axvline(x=value, color='k', linestyle='--', lw=2)
plt.show()