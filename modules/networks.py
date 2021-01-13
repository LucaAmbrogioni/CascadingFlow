import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TriResNet(nn.Module):

    def __init__(self, d_x, d_epsilon, epsilon_nu, in_pre_lambda=None, scale_w=0.01):
        super(TriResNet, self).__init__()
        self.d_x = d_x
        self.d_epsilon = d_epsilon
        self.width = d_x + d_epsilon
        self.epsilon_nu = epsilon_nu
        self.in_pre_lambda = in_pre_lambda

        # Learnable parameters
        self.W1 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,self.width))))
        self.W2 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,self.width))))
        self.W3 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,self.width))))
        self.d1 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,))))
        self.d2 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,))))
        self.d3 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,))))
        self.b1 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,))))
        self.b2 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,))))
        self.b3 = nn.Parameter(torch.Tensor(np.random.normal(0,scale_w,(self.width,))))
        self.eps_b = nn.Parameter(torch.Tensor(np.zeros((d_epsilon,))))
        self.eps_s = nn.Parameter(torch.Tensor(-4*np.ones((d_epsilon,))))
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

    # def inverse(self, y, epsilon_out):
    #     #Matrices
    #     d1, d2, d3, U1, L1, U2, L2, U3, L3 = self.get_matrices()
    #
    #     # Inverse
    #     z3 = torch.cat((y, epsilon_out),1)
    #     z2 = self.inv_f(self.inv_LU_layer(z3, L3, U3, self.b3))
    #     z1 = self.inv_f(self.inv_LU_layer(z2, L3, U3, self.b3))
    #     z0 = self.inv_LU_layer(z1, L1, U1, self.b1)
    #     x_out, epsilon_out = z0[:,:self.d_x], z3[:,self.d_x:]
    #     l = self.get_l()
    #     log_jacobian = torch.sum(torch.log(l + (1-l)*d1) + torch.log(l + (1-l)*d2) + torch.log(l + (1-l)*d3)) + torch.mean(torch.sum(self.log_df(z1) + self.log_df(z2),1))
    #     return x_out, epsilon_out, log_jacobian

    def __call__(self, x, eps_mean=None, eps_scale=None, local_eps=None):
        D = self.d_epsilon
        N = x.shape[0]
        m = eps_mean if eps_mean is not None else 0.
        s = eps_scale if eps_scale is not None else self.epsilon_nu
        if local_eps is not None:
            epsilon = local_eps
        else:
            epsilon = m + s*torch.distributions.normal.Normal(torch.zeros((N,D)),torch.ones((N,D))).rsample()
        x_posterior, epsilon_out, log_jacobian = self.forward(x, epsilon)
        out_scale = F.softplus(self.eps_s)
        epsilon_out = (epsilon_out - self.eps_b)*out_scale
        log_jacobian += torch.sum(torch.log(out_scale))
        #print(np.mean(epsilon_out.detach().numpy()))
        #print("std: {}".format(np.std(epsilon_out.detach().numpy())))
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

    def __init__(self, node_size, depth, in_scale, scale, in_w=1.8):
        super(LinearGaussianTree, self).__init__()
        self.node_size = node_size
        self.depth = depth
        self.in_scale = in_scale
        self.scale = scale
        self.size = 2**depth - 1
        self.weights = nn.Parameter(torch.tensor(np.random.normal(in_w,0.0001,(self.size,))))
        #self.scaling_factors = nn.Parameter(torch.tensor(np.ones((self.size,))))
        #self.bias_factors = nn.Parameter(torch.tensor(np.ones((self.size,))))

    def sample(self, M):
        samples_list = [torch.distributions.normal.Normal(0., self.in_scale).rsample((M,self.node_size,1))]#.type(torch.float32)]
        m_list = []
        scale_list = []
        tot_idx = 0
        for d in range(1,self.depth):
            samples_d_list = []
            m_d_list = []
            scale_d_list = []
            parent_half_idx = 0
            for j in range(2**d):
                w = torch.sigmoid(self.weights[tot_idx])
                #a = torch.sigmoid(self.scaling_factors[tot_idx])
                #b = self.bias_factors[tot_idx]
                m = w*samples_list[d-1][:, :, int(parent_half_idx)].unsqueeze(2)# + b
                scale = ((1-w)*self.scale).repeat(m.shape)
                new_sample = torch.distributions.normal.Normal(m,scale).rsample()
                parent_half_idx += 1/2
                tot_idx += 1
                samples_d_list.append(new_sample)
                m_d_list.append(m)
                scale_d_list.append(scale)
            samples_list.append(torch.cat(samples_d_list,2))
            m_list.append(torch.cat(m_d_list, 2))
            scale_list.append(torch.cat(scale_d_list, 2))
        return torch.cat(list(reversed(samples_list)),2), torch.cat(list(reversed(m_list)),2), torch.cat(list(reversed(scale_list)),2)


class LinearGaussianChain(nn.Module):

    def __init__(self, node_size, T, in_scale, amortized=False, data_dim=None):
        super(LinearGaussianChain, self).__init__()
        self.node_size = node_size
        self.T = T
        self.in_scale = in_scale
        self.is_amortized = amortized
        self.coupling_layers = nn.ModuleList()
        self.data_embedding_layers = nn.ModuleList()
        self.noise_embedding = nn.Parameter(torch.ones((1,node_size,T)))
        for _ in range(T):
            self.coupling_layers.append(nn.Linear(node_size, node_size))
            if amortized:
                self.data_embedding_layers.append(nn.Linear(data_dim, node_size))

    def sample(self, M, data): #TODO: Work in progress
        m0 = torch.zeros((M,self.node_size,1))
        s0 = self.in_scale*torch.ones((M,self.node_size,1))
        m_list = [m0]
        scale_list = [s0]
        samples_list = [torch.distributions.normal.Normal(m0, s0).rsample()]#.type(torch.float32)]
        for t in range(1,self.T):
            noise_sample = torch.distributions.normal.Normal(0,1.).sample((M,self.node_size))
            st = F.softplus(self.noise_embedding[:,:,t]).repeat((M,1))
            data_inpt = self.data_embedding_layers[t](data[self.T-t].view((M,1))) if (self.is_amortized and data[self.T-t] is not None) else 0.
            coupling_inpt = self.coupling_layers[t](samples_list[-1].view((M,self.node_size)))
            mt = data_inpt + coupling_inpt
            m_list.append(mt.view((M,self.node_size,1)))
            scale_list.append(st.view((M,self.node_size,1)))
            samples_list.append((mt + st*noise_sample).view((M,self.node_size,1)))
        return torch.cat(list(reversed(samples_list)),2), torch.cat(list(reversed(m_list)),2), torch.cat(list(reversed(scale_list)),2)