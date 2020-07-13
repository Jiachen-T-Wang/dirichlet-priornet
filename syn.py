import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns

from scipy.stats import multivariate_normal
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

import torch.distributions as dist
from mpl_toolkits import mplot3d
from torch.distributions.dirichlet import Dirichlet

from dpn import PriorNet


def gen_ood(mean_lst, size=3):
  ood_data = np.empty((1, 2))
  for mean in mean_lst:
    data = np.random.multivariate_normal(mean=mean, cov=np.identity(2)*0.01, size=size)
    ood_data = np.concatenate([ood_data, data])
  return ood_data


def plot_scatter(data, color):
  plt.scatter(data[:, 0], data[:, 1], color=color)


def synthe_data(sigma, size_in, size_ood, scale=1):

  data1 = np.random.multivariate_normal(mean=np.array([2, 0])*scale, 
                                        cov=np.identity(2)*sigma, size=size_in)
  data2 = np.random.multivariate_normal(mean=np.array([-1, np.sqrt(3)])*scale, 
                                        cov=np.identity(2)*sigma, size=size_in)
  data3 = np.random.multivariate_normal(mean=np.array([-1, -np.sqrt(3)])*scale, 
                                        cov=np.identity(2)*sigma, size=size_in)

  ood_data_x = np.random.uniform(-10, 10, size_ood)
  ood_data_y = np.random.uniform(-10, 10, size_ood)
  ood_data = np.array([ood_data_x, ood_data_y]).T

  def remove_in(ood_data, sigma):
    mtn1 = multivariate_normal(mean=[2, 0], cov=np.identity(2)*sigma)
    mtn2 = multivariate_normal(mean=[-1, np.sqrt(3)], cov=np.identity(2)*sigma)
    mtn3 = multivariate_normal(mean=[-1, -np.sqrt(3)], cov=np.identity(2)*sigma)
    threshold = (norm.pdf(3.1))
    ood_data = ood_data[mtn1.pdf(ood_data)<threshold]
    ood_data = ood_data[mtn2.pdf(ood_data)<threshold]
    ood_data = ood_data[mtn3.pdf(ood_data)<threshold]
    return ood_data

  ood_data = remove_in(ood_data, sigma)
  return data1, data2, data3, ood_data


def entropy(dist):
    su=0
    for p in dist:
        r= p/sum(dist)
        if (r==0):
            su+=0
        else:
            su+= -r*(np.log(r))
    return su/np.log(2)


def entropy_map(xi,yi,sigma,scale = 1):
  ent = np.zeros((xi.shape[0],yi.shape[0]))
  cov_mat = np.identity(2)*sigma
  for i in range(xi.shape[0]):
    for j in range(yi.shape[0]):
      set1 = multivariate_normal.pdf([xi[i],yi[j]], 
                                     mean = np.array([2, 0])*scale, cov = cov_mat)
      set2 = multivariate_normal.pdf([xi[i],yi[j]], 
                                     mean = np.array([-1, np.sqrt(3)])*scale, cov = cov_mat)
      set3 = multivariate_normal.pdf([xi[i],yi[j]], 
                                     mean = np.array([-1, -np.sqrt(3)])*scale, cov = cov_mat)
      alpha = np.array([set1,set2,set3])
      res = entropy(alpha)
      ent[i,j] = res
  return ent


 def DE_map(shape_size, xlo, xhi, ylo, yhi):
  shape_size = 50
  DE_plain = np.zeros((shape_size, shape_size))

  for i in range(shape_size):
    for j in range(shape_size):
      x = np.linspace(xlo, xhi, shape_size)[i]
      y = np.linspace(ylo, yhi, shape_size)[j]
      DE_plain[i, j]=network.diffenrential_entropy([x, y]).cpu().data.numpy()[0][0]

  return DE_plain


 def get_traindata(data1, data2, data3, ood_data):
  
	 def gen_onehot(size, category, eps, total_cat=3):
	    label = np.zeros((size, total_cat))
	    label[:, category] = np.ones(size)*10
	    for i in range(total_cat):
	      if i!=category:
	        label[:, i] = np.ones(size)
	    return label

	label1 = gen_onehot(size=len(data1), category=0, eps=0.001)
	label2 = gen_onehot(size=len(data2), category=1, eps=0.001)
	label3 = gen_onehot(size=len(data3), category=2, eps=0.001)

	ood_label = np.ones((len(ood_data), 3))

	train_X = np.concatenate([data1, data2, data3, ood_data])
	train_Y = np.concatenate([label1, label2, label3, ood_label])
	return train_X, train_Y


if __name__ == '__main__':
	sigma = 0.1
	data1, data2, data3, ood_data = synthe_data(sigma, 500, 1500)
	train_X, train_Y = get_traindata(data1, data2, data3, ood_data)
	network = PriorNet(2, 50)
	if torch.cuda.is_available(): network.cuda()
	optimizer = optim.Adam(network.parameters(), lr=0.001)

	network.fit(10, optimizer, train_X, train_Y)
	DE_plain = DE_map(50, -9, 9, -9, 9)
	ax = sns.heatmap(DE_plain.T, linewidth=0.5)
	ax.invert_yaxis()
	plt.show()