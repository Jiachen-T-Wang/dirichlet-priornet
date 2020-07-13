import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import multivariate_normal
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

import torch.distributions as dist
from torch.distributions.dirichlet import Dirichlet

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc

from dpn import PriorNet_CNN, DNN


def uncertainty_score(model, test_data, metrics):
  model.eval()
  score_lst = []
  for x in test_data:
    x = [x]
    with torch.no_grad():
      if metrics=='DE':
        score = model.diffenrential_entropy(x).data.cpu().numpy()[0][0]
      elif metrics=='MI':
        score = model.mutual_information(x).data.cpu().numpy()[0]
      elif metrics=='MAXP':
        score = model.max_prob(x).data.cpu().numpy()[0]
      elif metrics=='ENT':
        score = model.entropy(x).data.cpu().numpy()[0]
    score_lst.append(score)
  return score_lst


def get_ood_label_score(test_in_score, test_out_score):
  score = np.concatenate([test_in_score, test_out_score])
  label = np.concatenate((np.zeros(len(test_in_score)), np.ones(len(test_out_score))))
  return label, score
  
def get_misc_label_score(model, test_data, test_label, test_in_score):
  misc_label = np.zeros(len(test_label))

  for i in range(len(test_data)):
    x = test_data[i]
    x = [x]
    with torch.no_grad():
      pred = model.predict_alpha(x)
      pred_class = torch.max(pred[0], 0).indices
      misc_label[i] = 1-torch.eq(test_label[i], pred_class).sum().numpy()

  return misc_label, np.array(test_in_score)

def plot_roc(label, score, label_name):
  fpr, tpr, thresholds = roc_curve(label, score)
  plt.plot(fpr, tpr, label=label_name)
  plt.xlabel('False Positive')
  plt.ylabel('True Positive')
  plt.title('ROC')
  plt.ylim(0.0, 1.0)
  plt.xlim(0.0, 1.0)

def plot_pr(label, score, label_name):
  precision, recall, thresholds = precision_recall_curve(label, score)
  plt.plot(recall, precision, label=label_name)
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Precision-Recall Curve')
  plt.ylim(0.0, 1.0)
  plt.xlim(0.0, 1.0)


def get_auroc_aupr(label, score):
  #score[score==-np.inf] = -100
  #score[score==np.inf] = 100

  auroc = roc_auc_score(label, score)
  precision, recall, thresholds = precision_recall_curve(label, score)
  aupr = auc(recall, precision)
  return auroc, aupr


def get_test_roc_pr(network, metrics, detect='ood'):
  test_in_score = uncertainty_score(network, test_in.data.numpy(), metrics)
  test_out_score = uncertainty_score(network, test_out.data.numpy(), metrics)
  if detect=='ood':
    label_dp, score_dp = get_ood_label_score(test_in_score, test_out_score)
  elif detect=='misc':
    label_dp, score_dp = get_misc_label_score(network, test_in.data.numpy(),
                                              test_in.targets, test_in_score)
  if metrics=='MAXP':
    score_dp = -score_dp

  index = np.isposinf(score_dp)
  score_dp[np.isposinf(score_dp)] = 1e9
  maximum = np.amax(score_dp)
  score_dp[np.isposinf(score_dp)] = maximum + 1

  index = np.isneginf(score_dp)
  score_dp[np.isneginf(score_dp)] = -1e9
  minimum = np.amin(score_dp)
  score_dp[np.isneginf(score_dp)] = minimum - 1

  score_dp[np.isnan(score_dp)] = 0

  auroc, aupr = get_auroc_aupr(label_dp, score_dp)
  return auroc, aupr, label_dp, score_dp



if __name__=='__main__':

  train_in = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                        transform=torchvision.transforms.Compose(
                                            [torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,)),
                                             ]))

  test_in = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose(
                                            [torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,)),
                                             ]))
  train_out = torchvision.datasets.FashionMNIST('/files/', train=True, download=True,
                                        transform=torchvision.transforms.Compose(
                                            [torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))]))
  test_out = torchvision.datasets.FashionMNIST('/files/', train=False, download=True,
                                        transform=torchvision.transforms.Compose(
                                            [torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                          (0.1307,), (0.3081,))]))

  train_out.targets = torch.tensor(np.ones(len(train_out.targets))*10, dtype=torch.long)
  test_out.targets = torch.tensor(np.ones(len(test_out.targets))*10, dtype=torch.long)

  train_all = train_in
  train_all.data = torch.cat((train_in.data, train_out.data))
  train_all.targets = torch.cat((train_in.targets, train_out.targets))

  network = PriorNet_CNN()
  if torch.cuda.is_available(): network.cuda()
  optimizer = optim.Adam(network.parameters(), lr=0.001)

  network.fit(5, optimizer, train_all)

  auroc_mp, aupr_mp, label_mp, score_mp = get_test_roc_pr(dpn, 'MAXP', 'misc')
  auroc_ent, aupr_ent, label_ent, score_ent = get_test_roc_pr(dpn, 'ENT', 'misc')
  auroc_mi, aupr_mi, label_mi, score_mi = get_test_roc_pr(dpn, 'MI', 'misc')
  auroc_de, aupr_de, label_de, score_de = get_test_roc_pr(dpn, 'DE', 'misc')






