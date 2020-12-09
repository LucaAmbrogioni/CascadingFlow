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