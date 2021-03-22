import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzTransition():

    def __init__(self, dt, sigma=10, rho=28, beta=8 / 3):
        self.dt = dt
        self.s = sigma
        self.r = rho
        self.b = beta

    def __call__(self, inpt, mu):
        x, y, z = torch.split(inpt, split_size_or_sections=1, dim=1)
        return inpt + self.dt * torch.cat((self.s * (y - x), x * (self.r - z) - y, x * y - self.b * z), 1)


class VanDerPollTransition():

    def __init__(self, dt, mu=2., omega=5 * 2 * np.pi):
        self.dt = dt
        self.mu = mu
        self.omega = omega

    def __call__(self, inpt, mu):
        x, v = torch.split(inpt, split_size_or_sections=1, dim=1)
        return inpt + self.dt * torch.cat((v,
                                           self.mu * (1 - x ** 2) * v - self.omega * x), 1)

    # def __call__(self, inpt, mu):
    #  f = lambda x,y: (v, self.mu*(1 - x**2)*v - self.omega*x)
    #  h = 0.5*self.dt
    #  x,v = torch.split(inpt, split_size_or_sections=1, dim=1)
    #  x1,v1 = f(x,v)
    #  x2,v2 = f(x + h*x1, v + h*v1)
    #  x3, v3 = f(x + h*x2, v + h*v2)
    #  return inpt + self.dt*torch.cat(f(x + self.dt * x3, v + self.dt * v3),1)


class VolterraTransition():

    def __init__(self, dt, a=0.2, b=0.02, c=0.02, d=0.1):
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, inpt, mu):
        x, y = torch.split(inpt, split_size_or_sections=1, dim=1)
        return torch.relu(inpt + self.dt * torch.cat((self.a * x - self.b * x * y,
                                                      self.c * x * y - self.d * y), 1))


class BrusselatorTransition():

    def __init__(self, dt, a=1, b=3):
        self.dt = dt
        self.a = a
        self.b = b

    def __call__(self, inpt, mu):
        x, y = torch.split(inpt, split_size_or_sections=1, dim=1)
        return inpt + self.dt * torch.cat((self.a + x ** 2 * y - self.b * x - x,
                                           self.b * x - x ** 2 * y), 1)


class RNNTransition():

    def __init__(self, d_x, d_h, dt, s=1.):
        self.dt = dt
        self.W1 = torch.tensor(np.random.normal(0., s, (d_h, d_x))).float()
        self.W2 = torch.tensor(np.random.normal(0., s, (d_h, d_h))).float()
        self.W3 = torch.tensor(np.random.normal(0., s, (d_x, d_h))).float()
        self.b1 = torch.tensor(np.random.normal(0., s, (d_h,))).float()
        self.b2 = torch.tensor(np.random.normal(0., s, (d_h,))).float()

    def __call__(self, inpt, mu):
        h = torch.tanh(torch.nn.functional.linear(inpt, self.W1) + self.b1)
        h = torch.tanh(torch.nn.functional.linear(h, self.W2) + self.b2)
        return inpt + self.dt * torch.tanh(torch.nn.functional.linear(h, self.W3))


class ConvTransition():
    def __init__(self, in_ch, out_ch, kernel_size, pad, dt, s=1., activation=F.relu):
        # padding must be one for mnist
        self.dt = dt
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv.weight = nn.Parameter(
            torch.tensor(np.random.normal(0., s, (out_ch, in_ch, kernel_size, kernel_size))), requires_grad=False)
        self.activation_fn = activation

    def __call__(self, inpt, mu):
        h = self.conv(inpt)
        if self.activation_fn:
            h = self.activation_fn(h)
        return inpt + self.dt * h
