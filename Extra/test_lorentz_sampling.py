import numpy as np
import matplotlib.pyplot as plt

from modules.distributions import NormalDistribution, BernoulliDistribution
from modules.models import DynamicModel
from modules.dynamics import LorentzTransition
from modules.emissions import SingleCoordinateEmission


T = 40
dt = 0.02
sigma = np.sqrt(dt)*2.
initial_sigma = 1.
observation_gain = 2.
d_x = 3
dist = NormalDistribution()
bm = DynamicModel(sigma=sigma, initial_sigma=initial_sigma, distribution=dist, d_x=d_x,
                  transition=LorentzTransition(dt=dt),
                  emission=SingleCoordinateEmission(k=0, gain=observation_gain),
                  emission_distribution=BernoulliDistribution(), observation_gain=observation_gain, T=T)
N = 12
bm_sample,_ ,_ ,_ ,_ ,_ = bm.sample_timeseries(N)

plt.plot(np.transpose(bm_sample[:,0,:].detach().numpy()))
plt.show()

X_true, Y, mu =  bm.sample_observations(N)

print(1.)

#bm_obs_sample = bm.sample_observations(1)
x = X_true[0, 0, :].detach().numpy()
y = Y[0, :].detach().numpy()
plt.scatter(range(T), y)
plt.plot((x - np.min(x))/np.max(x- np.min(x)))
plt.show()