import re
from copy import copy
import random
import os
import sys

#sys.stderr = object
#sys.tracebacklimit=0
import torch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.special import logit, expit
from scipy.stats import norm, rv_continuous, entropy, wasserstein_distance
from sklearn.neighbors import KernelDensity
from scipy.stats.kde import gaussian_kde


def get_thetas(states):
   thetas = np.arctan2(states[:,1], states[:,0])
   return thetas


def plot_states(thetas_train, actions_train, thetas_test, actions_test, store_dir, suffix):
    plt.scatter(thetas_train, actions_train, marker='x', label = 'train', alpha = 0.05, s=10)
    plt.scatter(thetas_test, actions_test, marker='o', label='test', alpha = 0.05, s=10)
    plt.legend(loc='upper left')
    file_name = os.path.join(store_dir, (f'thetavthetadot_'+suffix+'.png'))
    plt.savefig(file_name)
    plt.close()

def plot_likelihood_histograms(likelihoods, suffix, train_dataset, 
    store_dir = '', show = False):
    #width = len(likelihoods)*12
    #f, axes = plt.subplots(1, len(likelihoods), figsize=(width,12))
    #f, axes = plt.subplots(1, 1)
    colors = ['skyblue', 'lightcoral', 'limegreen', 'mediumpurple', 'bisque', 'sandybrown']
    index = 1
    for key, likelihood in likelihoods.items():
        likelihoods[key] = likelihoods[key].numpy()
    sns.histplot(likelihoods, kde = True, stat = 'density', log_scale=(False, False))
    #for key, likelihood in likelihoods.items():
    #    import pdb; pdb.set_trace()
    #    sns.histplot(likelihood.numpy(), kde = False, stat = 'density',
    #        label = key) #color=colors[index], edgecolor = 'black')
#            ax = axes[index])
    #    index += 1
    plt.legend()
    # title
    plt.title(f'Trained on {train_dataset}')
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('loglikeli_hist_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_state1(samp, data, show = False, suffix='jackie', store_dir='robinson', kde=True):
    width = samp.shape[1]*12
    f, axes = plt.subplots(1, samp.shape[1], sharey=False, figsize=(width,12))
    f.suptitle("Histogram of Data", fontsize=40)
    for i in range(data.shape[1]):
        # bins = 60
        if samp.shape[1] == 1:
            plot_ax = axes
        else:
            plot_ax = axes[i]
        sns.histplot(data[:,i], kde=kde, color = 'skyblue',
             edgecolor = 'black', stat = 'count',
             label = 'ground_truth', ax = plot_ax)
        sns.histplot(samp[:,i], kde=True, color = 'red',
             edgecolor = 'black', stat = 'count', 
             label = 'sampled_model', ax = plot_ax)
        plot_ax.legend(loc = 'upper right')
        plot_ax.set_title(f'Dimension {i}', fontsize=20)
    f.text(0.5, 0.04, 'Position', ha='center', fontsize=30)
    f.text(0.04, 0.5, 'Normalized Frequency', va='center', rotation='vertical', fontsize=30)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('state_distribution_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

def plot_rmse_likelihood(rmse_likelihood, x, label, show=False, store_dir='robinson'):
    plt.plot(x, rmse_likelihood)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, (f'{label}.png'))
        plt.savefig(file_name)
        plt.close()

def plot_total_ent(ent, x, store_dir, suffix, show=False):
    plt.plot(x, ent)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, (suffix+'.png'))
        plt.savefig(file_name)
        plt.close()


def plot_uncertainty_1d(mutual_info, kl_exp, bhatt_exp, x_cord, store_dir, suffix, show=False):
    f, axes = plt.subplots(1, 3, figsize=(45,15))
    axes[0].plot(x_cord, mutual_info.reshape(x_cord.shape))
    axes[0].set_title('MC Uncertainty')
    axes[1].plot(x_cord, kl_exp.reshape(x_cord.shape))
    axes[1].set_title('KL Uncertainty')
    axes[2].plot(x_cord, bhatt_exp.reshape(x_cord.shape))
    axes[2].set_title('Bhatt Uncertainty')
    epoch = re.search(r'epoch_[0-9]+', suffix).group()
    epoch = epoch.split('_')[1]
    f.suptitle(f'Epoch {epoch}', fontsize=26)
    if show:
        plt.show()
    else:
        file_name = os.path.join(store_dir, ('dep_uncertainty_'+suffix+'.png'))
        plt.savefig(file_name)
        plt.close()

