import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LorentzTransition():

  def __init__(self, dt, sigma=10, rho=28, beta=8/3):
    self.dt = dt
    self.s = sigma
    self.r = rho
    self.b = beta

  def __call__(self, inpt, mu):
    x,y,z = torch.split(inpt, split_size_or_sections=1, dim=1)
    return inpt + self.dt*torch.cat((self.s*(y - x), x*(self.r - z)-y,x*y - self.b*z),1)

class VolterraTransition():

  def __init__(self, dt, a=0.2, b=0.02, c=0.02, d=0.1):
    self.dt = dt
    self.a = a
    self.b = b
    self.c = c
    self.d = d

  def __call__(self, inpt, mu):
    x,y = torch.split(inpt, split_size_or_sections=1, dim=1)
    return torch.relu(inpt + self.dt*torch.cat((self.a*x - self.b*x*y,
                                                self.c*x*y - self.d*y),1))

class BrusselatorTransition():

  def __init__(self, dt, a=1, b=3):
    self.dt = dt
    self.a = a
    self.b = b

  def __call__(self, inpt, mu):
    x,y = torch.split(inpt, split_size_or_sections=1, dim=1)
    return inpt + self.dt*torch.cat((self.a + x**2*y - self.b*x - x,
                                     self.b*x - x**2*y),1)


class RNNTransition():

  def __init__(self, d_x, d_h, dt, s=1.):
    self.dt = dt
    self.W1 = torch.tensor(np.random.normal(0.,s,(d_h,d_x))).float()
    self.W2 = torch.tensor(np.random.normal(0.,s,(d_h,d_h))).float()
    self.W3 = torch.tensor(np.random.normal(0.,s,(d_x,d_h))).float()
    self.b1 = torch.tensor(np.random.normal(0.,s,(d_h,))).float()
    self.b2 = torch.tensor(np.random.normal(0.,s,(d_h,))).float()

  def __call__(self, inpt, mu):
    h = torch.tanh(torch.nn.functional.linear(inpt, self.W1) + self.b1)
    h = torch.tanh(torch.nn.functional.linear(h, self.W2) + self.b2)
    return inpt + self.dt*torch.tanh(torch.nn.functional.linear(h, self.W3))