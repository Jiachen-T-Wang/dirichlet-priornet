import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import torch.distributions as dist
from mpl_toolkits import mplot3d
from torch.distributions.dirichlet import Dirichlet

from scipy.stats import multivariate_normal
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys


def plot_dir(alpha, size):
  model = Dirichlet(torch.tensor(alpha))
  sample = model.sample(torch.Size([size])).data
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.scatter3D(sample[:, 0], sample[:, 1], sample[:, 2], color='red')
  ax.plot([0, 0], [1, 0], [0, 1], linewidth=3, color='purple')
  ax.plot([0, 1], [0, 0], [1, 0], linewidth=3, color='purple')
  ax.plot([0, 1], [1, 0], [0, 0], linewidth=3, color='purple')
  ax.set_xlim((0, 1))
  ax.set_ylim((0, 1))
  ax.set_zlim((0, 1))
  ax.view_init(60, 35)


def numpy_to_tensor(arr):
  ret = torch.Tensor([arr])
  return ret