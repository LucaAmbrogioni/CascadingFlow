from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from modules.distributions import NormalDistribution, BernoulliDistribution
from modules.models import HierarchicalModel
from modules.networks import TriResNet

mean_sigma = 1.
scale_mu = 0.2
scale_sigma = 0.05
n_children = 10
emission_sigma_list = [0.5 for _ in range(n_children)]
d_x = 1
mean_dist = NormalDistribution()
scale_dist = NormalDistribution()
children_dist= NormalDistribution()
mean_link = lambda x: x
scale_link = lambda x: torch.exp(x)
emission = lambda x, r: x
emission_distribution = NormalDistribution()

model = HierarchicalModel(n_children, d_x, mean_sigma, scale_mu, scale_sigma, emission_sigma_list, mean_dist, scale_dist,
                          children_dist, mean_link, scale_link, emission, emission_distribution)
N = 40
M = 500
_, y  = model.sample_hierarchical_observations(N, M)

#for n in range(n_children):
#    plt.hist(y[n][:,0].detach().numpy().flatten(), 25, alpha=0.5)
#plt.show()

num_repetitions = 1
num_iterations = 1000
batch_size = 50
prior_model = model
M = 5

for rep in range(num_repetitions):

    print("Repetition: {}".format(rep))

    # generate ground truth
    X_true, Y =  prior_model.sample_hierarchical_observations(1, M)
    data = Y


    ### Cscading flowsd ###
    transformations = [TriResNet(d_x=d_x, d_epsilon=20, epsilon_nu=0.05, in_pre_lambda=1.) for _ in
                       range(2 + n_children)]
    variational_model = HierarchicalModel(n_children, d_x, mean_sigma, scale_mu, scale_sigma, emission_sigma_list, mean_dist,
                                          scale_dist,
                                          children_dist, mean_link, scale_link, emission, emission_distribution,
                                          transformations=transformations)
    loss_list_cf = []
    parames_list = [tr.parameters() for tr in transformations]
    params = []
    for p in parames_list:
        params += p
    optimizer = optim.Adam(params, lr=0.001)

    print("Train Cascadinf Flow model")
    for itr in tqdm(range(num_iterations)):
        # Gradient reset
        optimizer.zero_grad()

        # Variational loss
        X, x_pre, log_jacobian, epsilon_loss, _ = variational_model.sample(batch_size)
        log_q = variational_model.evaluate_avg_joint_log_prob(X, None, x_pre=x_pre, log_jacobian=log_jacobian,
                                                              epsilon_loss=epsilon_loss)
        log_p = prior_model.evaluate_avg_joint_log_prob(X, data)
        loss = (log_q - log_p)

        # Update
        loss.backward()
        optimizer.step()
        loss_list_cf.append(float(loss.detach().numpy()))

    # Print results
    print("Parents:")
    print("True parent mean: {:.3}, Estimation: {:.3} +- {:.3}".format(float(X_true["mean"].detach().numpy()),
                                                                       float(np.mean(X["mean"].detach().numpy())),
                                                                       float(np.std(X["mean"].detach().numpy()))))
    print("True parent scale: {:.3}, Estimation: {:.3} +- {:.3}".format(float((X_true["scale"].detach().numpy())),
                                                                        float(np.mean((X["scale"].detach().numpy()))),
                                                                        float(np.std((X["scale"].detach().numpy())))))
    print("Children:")
    for c_true, c in zip(X_true["children"], X["children"]):
        print("True mean: {:.3}, Estimation: {:.3} +- {:.3}".format(float(c_true.detach().numpy()),
                                                                    float(np.mean(c.detach().numpy())),
                                                                    float(np.std(c.detach().numpy()))))

    # Plot
    plt.plot(loss_list_cf)
    plt.show()