import gc
from copy import copy
import random
import os
import sys
import itertools
from time import time
from datetime import timedelta

import torch
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.special import logit
from scipy.stats import dirichlet, moment, ks_2samp, anderson_ksamp
from scipy.stats import norm, rv_continuous, entropy, wasserstein_distance
from scipy.special import softmax
from sklearn.neighbors import KernelDensity

from plot_functions import plot_states, plot_state1
from utils import kl_mvn, Wasserstein_GP, entropy_utils
from envs_1d import hetero_samp_condition, bimodal_samp_condition

def get_thetas(states):
   thetas = np.arctan2(states[:,1], states[:,0])
   return thetas


def check_cuda_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return f
    


def check_dyna_fit_1d(env, model, replay_buffer, device, suffix, 
        store_dir, input_preproc, output_postproc, 
        plot=True, state='random', inp_stats=None, out_stats=None):
    x = np.random.choice(replay_buffer[0])
    x = x.reshape(-1,1)
    x = torch.tensor(x, dtype = torch.float32).to(device)
    x = input_preproc(x, model.stats_inputs)
    samp = model.sample(10000, context = x.to(device))
    samp = samp[0]
    samp = samp.squeeze(0)
    samp = output_postproc(samp, model.stats_outputs)
    samp = samp.detach().cpu()
    x = x.cpu()
    if env == 'bimodal':
        gt = bimodal_samp_condition(10000, x)
    else:
        gt = hetero_samp_condition(10000,x)
    gt = numpy.expand_dims(gt, axis=1)
    plot_state1(samp, gt, show=False, suffix=suffix, store_dir=store_dir, kde=True)



def calc_rmse(test_data, input_preproc, output_postproc, 
        model, ensemble_size =10, device = 'cuda'):
    batch = [tpl for tpl in test_data.buffer]
    states, actions, reward, next_states, done, noisy_actions, index = map(np.stack, zip(*batch))
    states = torch.tensor(states, dtype = torch.float32).to(device)
    actions = torch.tensor(actions, dtype = torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype = torch.float32).to(device)
    inps = torch.hstack([states, actions])
    inps = input_preproc(inps, model.stats_inputs)
    samp = []
    with torch.no_grad():
        for j in range(5):
            for i in range(ensemble_size):
                kwargs = {'rand_mask': False, 'mask_index': i}
                #comp_samp = model.sample(int((numb_samps==i).sum()), context = inps, kwargs=kwargs)
                comp_samp = model.sample(50, context = inps, kwargs=kwargs)
                samp += [comp_samp[0].detach().cpu()]
                del comp_samp
                gc.collect()
                torch.cuda.empty_cache()
    samp = torch.hstack(samp)
    samp = samp.squeeze()
    rmses = []
    for i in range(next_states.shape[0]):
        y_hat = output_postproc(samp[i,:], [i.cpu() for i in model.stats_outputs])
        y_gt = next_states[i,:].cpu()
        rmse_pt = torch.sqrt(((y_gt - y_hat.mean(0))**2).mean())
        #rmse_pt = torch.sqrt(((y_gt - y_hat)**2).mean())
        #rmse_pt = ((y_gt - y_hat.mean(0))**2).mean()
        rmses.append(rmse_pt.item())
    rmse_mean = np.nanmean(rmses)
    return rmse_mean

def calc_rmse_1d(test_data, input_preproc, output_postproc, 
        model, ensemble_size =10, device = 'cuda'):
    X = test_data[0].reshape(-1,1)
    X = torch.tensor(X, dtype = torch.float32).to(device)
    X = input_preproc(X, model.stats_inputs)
    y = test_data[1]
    numb_samps = np.random.choice(ensemble_size, size=500)
    samp = []
    torch.cuda.empty_cache()
    for i in range(ensemble_size):
        kwargs = {'rand_mask': False, 'mask_index': i}
        comp_samp = model.sample(int((numb_samps==i).sum()), context = X, kwargs=kwargs)
        samp += [comp_samp[0].detach().cpu()]
        del comp_samp
        gc.collect()
        torch.cuda.empty_cache()
    samp = torch.hstack(samp)
    samp = samp.squeeze()
    rmses = []
    for i in range(y.shape[0]):
        y_hat = output_postproc(samp[i,:].reshape(-1,1), model.stats_outputs)
        y_gt = y[i]
        rmse_pt = (torch.sqrt((y_gt - y_hat.mean())**2))
        rmses.append(rmse_pt.item())
    rmse_mean = np.mean(rmses)
    return rmse_mean
