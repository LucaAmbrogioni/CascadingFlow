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

def plot_img_model(variational_model, X_true, M, savename=None):
    X, mu, x_pre, log_jacobian, epsilon_loss = variational_model.sample_timeseries(M)
    X = X.permute(0,4,1,2,3)
    plot_img_steps(X.shape[1], X[0].detach().numpy(), title='Variational')
    plot_img_steps(X.shape[1], X_true[0].detach().numpy(), title='True')

    '''if savename is None:
        plt.show()
    else:
        plt.savefig(savename + "_{}".format(k))
        plt.clf()'''

def plot_img_steps(T, X, title=None):
    # X shape: (TimeSteps, Channels, Width, Height)
    fig = plt.figure()
    for i in range(T):
        plt.subplot(1, T, i + 1)
        plt.tight_layout()
        plt.imshow(X[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    if title:
        fig.suptitle(title, fontsize=14)
    plt.show()