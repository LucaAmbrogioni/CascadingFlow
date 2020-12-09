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
    N = inpt.shape[0]
    x,y,z = torch.split(inpt, split_size_or_sections=1, dim=1)
    return inpt + self.dt*torch.cat((self.s*(y - x), x*(self.r - z)-y,x*y - self.b*z),1)