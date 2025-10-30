import os
import itertools
from time import time
from datetime import timedelta

import torch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

from plot_functions import plot_uncertainty_1d   
from utils import kl_mvn, Wasserstein_GP, entropy_utils
from uncertainty_estimator import EpistemicUncertaintyEstimator
from badge_bait.badge_sampling import BadgeSampling
from badge_bait.bait_sampling import BaitSampling


def init_centers(embs, K):
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    embs = embs.cuda()
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
        else:
            newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
            for i in range(len(embs)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll



def find_best_points(samp, numb_points, model, input_preproc, 
        ensemble_size, device, nflows=False, 
        acquisition_criteria = 'mutual_info', numb_points_2_add = 10, 
        x_train = None, y_train = None):
    state = samp[0]
    action = samp[1]
    model_inp = np.hstack([state, action])
    model_inp = torch.tensor(model_inp).type(torch.float32).to(device)
    model_inp = input_preproc(model_inp, model.stats_inputs)
    eue = EpistemicUncertaintyEstimator(acquisition_criteria)
    uncertainty, time_taken = estimate_uncertainty(model_inp, acquisition_criteria, model, 
        ensemble_size, numb_samps = numb_points, epi_estimator = eue, 
        estimator_types = [], nflows=nflows, x_train=x_train, y_train=y_train)
    if 'selected_indices' not in uncertainty.keys():
        uncertainty = uncertainty[acquisition_criteria]
        ind = np.argpartition(uncertainty, -numb_points_2_add)[-numb_points_2_add:]
    else:
        ind = uncertainty['selected_indices']
    points_2_add = []
    for i in ind:
        point = tuple((s[i] for s in samp))
        points_2_add.append(point)
    return points_2_add, time_taken


def find_best_points_1d(samp, numb_points, model, 
        input_preproc, ensemble_size, device, nflows = False,
        acquisition_criteria = 'mutual_info', numb_points_2_add = 10,
        x_train = None, y_train = None):
    X = samp[0].reshape(-1,1)
    X = torch.tensor(X, dtype = torch.float32).to(device)
    X = input_preproc(X, model.stats_inputs)
    eue = EpistemicUncertaintyEstimator(acquisition_criteria)
    uncertainty, time_taken = estimate_uncertainty(X, acquisition_criteria, model, 
        ensemble_size, numb_samps = numb_points, epi_estimator = eue, 
        estimator_types = [], nflows=nflows, x_train=x_train, y_train=y_train)
    if 'selected_indices' in uncertainty.keys():
        points_2_add = (samp[0][uncertainty['selected_indices']], samp[1][uncertainty['selected_indices']])
    else:
        uncertainty = uncertainty[acquisition_criteria]
        ind = np.argpartition(uncertainty, -numb_points_2_add)[-numb_points_2_add:]
        points_2_add = (samp[0][ind], samp[1][ind])
    return points_2_add, time_taken


def estimate_uncertainty(model_inp, uncertainty_type, model, 
        ensemble_size, numb_samps=10000, nflows=False, epi_estimator=[], estimator_types=[],
        acquisition_batch_size=10, x_train=None, y_train=None):
    uncertainty = {}
    if uncertainty_type in ['all', 'sample_bald', 'alea_unc', 'tot_unc', 'batchbald']:
        means = []
        stds = []
        output = []
        log_probs_base = []
        component_ent = []
        dep_t0 = time() 
        mut_info = []
        chunk_size = numb_samps
        if numb_samps > 100:
            chunk_size = 50
        for i in range(ensemble_size):
            kwargs = {'rand_mask': False, 'mask_index': i}
            chunks = int(np.ceil(numb_samps/chunk_size))
            output_hat = []
            log_prob_base = []
            for j in range(chunks):
                (output_hat_ch, log_prob_ch, base_mean, base_std) = (
                    model.sample_and_log_prob(chunk_size, context = model_inp, 
                    kwargs=kwargs, ensemble = True, ensemble_size = ensemble_size))
                output_hat.append(output_hat_ch.detach().cpu().numpy())
                log_prob_base.append(log_prob_ch.detach().cpu().numpy())
            output_hat = np.hstack(output_hat)
            mu = base_mean[::chunk_size]
            sig = base_std[::chunk_size]
            mu = mu.detach().cpu().numpy()
            sig = sig.detach().cpu().numpy()
            output.append(output_hat)
            if nflows:
                log_prob_base = np.concatenate(log_prob_base, axis=2)
                log_probs_base.append(log_prob_base)
            ent = mu.shape[1]/2*np.log(2*np.pi*np.e)+1/2*np.log((sig**2)).sum(1)
            component_ent.append(ent)
            means.append(mu)
            stds.append(sig)
        
        output = np.hstack(output)
        output = np.transpose(output, (1, 0, 2))
        alea_unc = np.stack(component_ent).mean(0)
        if not nflows: 
            probs = []
            for i in range(ensemble_size):
                norm_rv = torch.distributions.normal.Normal(torch.tensor(means[i]), 
                    torch.tensor(stds[i]))
                prob_comp = 1/ensemble_size*torch.exp(norm_rv.log_prob(torch.tensor(output)).sum(2))
                probs.append(prob_comp)
            probs = torch.stack(probs)
            probs = probs.numpy()
            log_probs = np.log(probs.sum(0))
        else:
            log_probs_base = np.concatenate(log_probs_base, axis=2)
            log_probs_base = np.transpose(log_probs_base, (0, 2, 1))
            probs = np.exp(log_probs_base)
            log_probs = np.log(probs.mean(0))
        if uncertainty_type == 'batchbald':
            selected_probs = []
            selected_indices = []
            alea_unc = np.stack(component_ent).mean(0)
            for j in range(acquisition_batch_size):
                cur_batch_size = len(selected_indices) 
                if cur_batch_size == 0:
                    total_unc = -numpy.ma.masked_invalid(log_probs).mean(0).data
                    epi_unc = total_unc - alea_unc 
                else:
                    prod_probs = np.transpose(np.stack(selected_probs), (1,2,0))
                    prod_probs = prod_probs.prod(2)
                    log_probs = np.log((np.transpose(probs, (2,0,1))*prod_probs).sum(1))
                    total_unc = -numpy.ma.masked_invalid(log_probs).mean(1).data
                    epi_unc = total_unc - alea_unc 
                index = np.nanargmax(epi_unc)
                selected_indices.append(index)
                selected_probs.append(probs[:,:,index])
                probs = np.delete(probs, index, axis=2)
                alea_unc = np.delete(alea_unc, index, axis=0)
            uncertainty['total_unc'] = total_unc
            uncertainty['batchbald'] = epi_unc
            uncertainty['alea_unc'] = alea_unc
            uncertainty['selected_indices'] = selected_indices 
        if uncertainty_type == 'sample_bald' or uncertainty_type =='all':
            total_unc = -numpy.ma.masked_invalid(log_probs).mean(0).data
            uncertainty['total_unc'] = total_unc
            epi_unc = total_unc-alea_unc
            uncertainty['sample_bald'] = epi_unc
            uncertainty['alea_unc'] = alea_unc
        dep_t1 = time()
        seconds_taken = dep_t1-dep_t0
    paides_list = ['kl_exp', 'bhatt_exp', 'wasserstein_exp']
    if uncertainty_type=='all' or uncertainty_type in paides_list:
        means = []
        stds = []
        pairwise_t0 = time()
        numb_samps = 1
        ents = []
        for i in range(ensemble_size):
            kwargs = {'rand_mask': False, 'mask_index': i}
            _, _, mu, sig = model.model.sample(numb_samps, 
                context = model_inp, kwargs=kwargs)
            mu = mu.detach().cpu().numpy()
            sig = sig.detach().cpu().numpy()
            ent = mu.shape[1]/2*np.log(2*np.pi*np.e)+1/2*np.log((sig**2).prod(1))
            ents.append(ent)
            means.append(mu)
            stds.append(sig)
        alea_unc = np.stack(ents).mean(0)
        mus = np.stack(means, axis=1)
        sigs = np.stack(stds, axis=1)
        mus = torch.tensor(mus)
        sigs = torch.tensor(sigs)
        weights = [1/ensemble_size for i in range(ensemble_size)]
        mi = epi_estimator.estimate_epi_uncertainty('', '', mus, sigs, 
                weights)
        uncertainty[epi_estimator.estimator] = mi
        uncertainty['alea_unc'] = alea_unc
        for et in estimator_types:
            mi = epi_estimator.estimate_epi_uncertainty('', '', mus, sigs, 
                    weights, method = et)
            uncertainty[et] = mi
        pairwise_t1 = time()
        seconds_taken = pairwise_t1-pairwise_t0
    if uncertainty_type == 'badge':
        pairwise_t0 = time()
        last_layer_gradients = model.grad_last_layer(model_inp)
        selected_indices = BadgeSampling(last_layer_gradients, acquisition_batch_size)
        uncertainty['selected_indices']=selected_indices
        pairwise_t1 = time()
        seconds_taken = pairwise_t1-pairwise_t0
    if uncertainty_type == 'bait':
        pairwise_t0 = time()
        last_layer_gradients = model.grad_last_layer(model_inp, num_samps=10, bait=True)
        if type(x_train) == np.ndarray:
            x_train = torch.tensor(x_train.astype(np.float32)).to(model_inp.device)
        last_layer_gradients_train = model.grad_last_layer(x_train, num_samps=10, bait=True)
        selected_indices = BaitSampling(last_layer_gradients.to('cpu'), 
            last_layer_gradients_train.to('cpu'), 
            acquisition_batch_size, x_train.shape[0])
        uncertainty['selected_indices']=selected_indices
        pairwise_t1 = time()
        seconds_taken = pairwise_t1-pairwise_t0
    print(f'{uncertainty_type} time: {str(timedelta(seconds=seconds_taken))}')
    return uncertainty, seconds_taken
