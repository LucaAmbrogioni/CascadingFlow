import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SingleCoordinateEmission():

  def __init__(self, k, gain):
    self.k = k
    self.gain = gain

  def __call__(self, x, r=None):
    return x[:,self.k]*self.gain


class LinearEmission():

  def __init__(self, in_feature, out_features):
    self.linear = nn.Linear(in_feature, out_features, bias=False)
    for param in self.linear.parameters():
      param.requires_grad = False

  def __call__(self, x, r=None):
    return self.linear(torch.flatten(x.permute(0,4,1,2,3), 2))
