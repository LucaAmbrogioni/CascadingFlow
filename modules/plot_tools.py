import numpy as np
import matplotlib.pyplot as plt

def plot_model(variational_model, X_true, K, M, savename=None):
    for k in range(K):
        X, mu, x_pre, log_jacobian, epsilon_loss = variational_model.sample_timeseries(M)
        plt.plot(np.transpose(X[:, k, :].detach().numpy()), alpha=0.2)
        plt.plot(np.mean(np.transpose(X[:, k, :].detach().numpy()), 1), c="r", lw="3", ls="--")
        plt.plot(np.transpose(X_true[0, k, :].detach().numpy()), c="k", lw="5", ls="--")
        if savename is None:
            plt.show()
        else:
            plt.savefig(savename + "_{}".format(k))
            plt.clf()
