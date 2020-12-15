import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TriResNet(nn.Module):

    def __init__(self, d_x, d_epsilon, epsilon_nu, in_pre_lambda=None):
        super(TriResNet, self).__init__()
        self.d_x = d_x
        self.d_epsilon = d_epsilon
        self.width = d_x + d_epsilon
        self.epsilon_nu = epsilon_nu
        self.in_pre_lambda = in_pre_lambda

        # Learnable parameters
        self.W1 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,self.width))))
        self.W2 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,self.width))))
        self.W3 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,self.width))))
        self.d1 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,))))
        self.d2 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,))))
        self.d3 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,))))
        self.b1 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,))))
        self.b2 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,))))
        self.b3 = nn.Parameter(torch.Tensor(np.random.normal(0,0.01,(self.width,))))
        if in_pre_lambda is None:
            self.pre_l = torch.Tensor(np.random.normal(-100., 0.1, (1,)))
        else:
            self.pre_l = nn.Parameter(torch.Tensor(np.random.normal(in_pre_lambda,0.1,(1,))))

        # Masks
        self.masku = torch.tril(torch.ones((self.width, self.width)), -1)
        self.maskl = torch.triu(torch.ones((self.width, self.width)), 1)

    def cu(self, M):
        I = torch.Tensor(np.identity(self.width))
        l = self.get_l().view((self.width,1))
        return l*I + (1 - l)*M

    def get_l(self, s=0.01):
        T = torch.cat((torch.Tensor(np.ones((self.d_x,))), torch.Tensor((np.zeros((self.d_epsilon,))))),0)
        return T*torch.sigmoid(self.pre_l) + s

    def log_df(self, x):
        return torch.log(self.df(x))

    def df(self, x):
        l = self.get_l()
        return l + (1 - l)*torch.sigmoid(x)*(1 - torch.sigmoid(x))

    def f(self, x):
        l = self.get_l()
        return l*x + (1 - l)*torch.sigmoid(x)

    def inv_f(self, y, x=0., N=20):
        x = torch.ones(y.shape)
        for _ in range(N):
          x = x - (self.f(x) - y)/(self.df(x))
        return x

    def LU_layer(self, x, L, U, b):
        x = torch.nn.functional.linear(x, self.cu(L))
        x = torch.nn.functional.linear(x, self.cu(U))
        return x + b

    def inv_LU_layer(self, x, L, U, b):
        x = torch.nn.functional.linear(x - b, torch.inverse(self.cu(U)))
        x = torch.nn.functional.linear(x, torch.inverse(self.cu(L)))
        return x

    def get_matrices(self, s = 0.01):
        d1 = s + F.softplus(self.d1)
        d2 = s + F.softplus(self.d2)
        d3 = s + F.softplus(self.d3)
        U1 = self.masku*self.W1 + torch.diag(d1)
        L1 = self.maskl*self.W1 + torch.Tensor(np.identity(self.width))
        U2 = self.masku*self.W2 + torch.diag(d2)
        L2 = self.maskl*self.W2 + torch.Tensor(np.identity(self.width))
        U3 = self.masku*self.W3 + torch.diag(d3)
        L3 = self.maskl*self.W3 + torch.Tensor(np.identity(self.width))
        return d1, d2, d3, U1, L1, U2, L2, U3, L3

    def forward(self, x, epsilon):
        #Matrices
        d1, d2, d3, U1, L1, U2, L2, U3, L3 = self.get_matrices()

        # Forward
        if len(x.shape)==1:
            x = x.view((x.shape[0],1))
        z0 = torch.cat((x, epsilon),1)
        z1 = self.LU_layer(z0, L1, U1, self.b1)
        z2 = self.LU_layer(self.f(z1), L2, U2, self.b2)
        z3 = self.LU_layer(self.f(z2), L3, U3, self.b3)
        x_out, epsilon_out = z3[:,:self.d_x], z3[:,self.d_x:]
        l = self.get_l()
        log_jacobian = torch.sum(torch.log(l + (1-l)*d1) + torch.log(l + (1-l)*d2) + torch.log(l + (1-l)*d3)) + torch.mean(torch.sum(self.log_df(z1) + self.log_df(z2),1))
        return x_out, epsilon_out, log_jacobian

    def inverse(self, y, epsilon_out):
        #Matrices
        d1, d2, d3, U1, L1, U2, L2, U3, L3 = self.get_matrices()

        # Inverse
        z3 = torch.cat((y, epsilon_out),1)
        z2 = self.inv_f(self.inv_LU_layer(z3, L3, U3, self.b3))
        z1 = self.inv_f(self.inv_LU_layer(z2, L3, U3, self.b3))
        z0 = self.inv_LU_layer(z1, L1, U1, self.b1)
        x_out, epsilon_out = z0[:,:self.d_x], z3[:,self.d_x:]
        l = self.get_l()
        log_jacobian = torch.sum(torch.log(l + (1-l)*d1) + torch.log(l + (1-l)*d2) + torch.log(l + (1-l)*d3)) + torch.mean(torch.sum(self.log_df(z1) + self.log_df(z2),1))
        return x_out, epsilon_out, log_jacobian

    def __call__(self, x, global_epsilon=0.):
        D = self.d_epsilon
        N = x.shape[0]
        epsilon = global_epsilon + torch.distributions.normal.Normal(torch.zeros((N,D)),self.epsilon_nu*torch.ones((N,D))).rsample()
        x_posterior, epsilon_out, log_jacobian = self.forward(x, epsilon)
        return x_posterior, epsilon, epsilon_out, log_jacobian

class LinearNet(nn.Module):

    def __init__(self, d_x):
        super(LinearNet, self).__init__()
        self.d_x = d_x
        self.l = nn.Linear(d_x, d_x)

    def __call__(self, x):
        return self.l(x)


class DeepNet(nn.Module):

    def __init__(self, d_x, d_h):
        super(DeepNet, self).__init__()
        self.d_x = d_x
        self.l1 = nn.Linear(d_x, d_h)
        self.l2 = nn.Linear(d_h, d_x)

    def __call__(self, x):
        return self.l2(F.relu(self.l1(x)))


class ASVIupdate(nn.Module):

  def __init__(self, d=1, l_init=1.):
    super(ASVIupdate, self).__init__()
    self.alpha_mu = nn.Parameter(torch.Tensor(np.random.normal(0.,0.1,(d,))))
    self.pre_lambda_mu = nn.Parameter(torch.Tensor(np.random.normal(l_init,0.1,(d,))))
    self.pre_alpha_s = nn.Parameter(torch.Tensor(np.random.normal(1.,0.1,(d,))))
    self.pre_lambda_s = nn.Parameter(torch.Tensor(np.random.normal(l_init,0.1,(d,))))

  def __call__(self, mu, s):
    lmu = torch.sigmoid(self.pre_lambda_mu)
    ls = torch.sigmoid(self.pre_lambda_s)
    new_mu = lmu*mu + (1 - lmu)*self.alpha_mu
    new_s = ls*s + (1 - ls)*F.softplus(self.pre_alpha_s)
    return new_mu, new_s


class LinearGaussianTree(nn.Module):

    def __init__(self, node_size, depth, in_scale, scale):
        super(LinearGaussianTree, self).__init__()
        self.node_size = node_size
        self.depth = depth
        self.in_scale = in_scale
        self.scale = scale
        self.size = 2**(depth+1) - 1
        self.weights = nn.Parameter(torch.tensor(np.random.normal(0.,0.1,(self.size,))))

    def sample(self, M):
        samples = torch.distributions.normal.Normal(0., self.in_scale).rsample((M,self.node_size,1))
        for d in range(1,self.depth):
            for j in range(2**d):
                parent_idx = 2**(d-1) - 1 + j//2
                w = torch.sigmoid(self.weights[parent_idx])
                m = w*samples[:, :, parent_idx].unsqueeze(2)
                new_sample = torch.distributions.normal.Normal(m,(1-w)*self.scale).rsample()
                samples = torch.cat((samples, new_sample),2)
        return samples