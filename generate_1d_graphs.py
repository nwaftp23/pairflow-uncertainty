import os 

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch import optim
import seaborn as sns
from tqdm import tqdm

from utils import normalize, un_normalize, seed_everything, identity
from envs_1d import hetero_samp, bimodal_samp, unimodal_samp
from nflows_ensemble_model import nflows_ensemble
from pens_model import pens
from mc_drop_model import mc_drop
from estimate_uncertainty import estimate_uncertainty
from uncertainty_estimator import EpistemicUncertaintyEstimator 

color_hexes = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', 
        '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

mpl.rc('font',family='Times New Roman')
data_path = './datasets/1d_envs'
graph_path = './graphs/'
# env can be bimodal or hetero
env = 'bimodal'
data_path = os.path.join(data_path,env)
os.makedirs(data_path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)
num_seeds = 10 
just_legend = False 
if just_legend:
    num_seeds = 2
train_data_all = []
samps_model = []
uncertainty_estimates = []
eue = EpistemicUncertaintyEstimator('kl_exp')
estimator_types = ['bhatt_exp']
# model_type can be any of pens, nflows_base or mc_dropout
model_type = 'pens'
numb_uncertainty_estimates = 100
numb_samples = 200

def export_legend(legend, filename="legend.png", expand=[-5,-5,5,5]):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join('graphs', filename), bbox_inches=bbox, dpi=400)

for i in tqdm(range(num_seeds)):
    tqdm.write(f'seed: {i}')
    seed_everything(i)
    if env == 'hetero':
        train_data = hetero_samp(numb_samples)
        datarange = np.linspace(-4.5, 4.5, numb_uncertainty_estimates)
    else:
        train_data = bimodal_samp(numb_samples)
        datarange = np.linspace(0, 3.5, numb_uncertainty_estimates)
    train_data_all.append(train_data)
    input_preproc = normalize
    output_preproc = normalize
    input_postproc = un_normalize
    output_postproc = un_normalize
    if model_type == 'nflows_base':
        num_layers = 1
        hids = 20
        ensemble_size = 5
        dropout_masks = True
        output_dim = 1
        context_dim = 1
        device = 'cpu'
        bins = 10
        domain = 1.2
        lr = 0.0005
        base = True
        epochs = 6000
        if just_legend:
            epochs = 200

        nflows_ensemble_model = nflows_ensemble(num_layers, hids, output_dim, context_dim,
                        bins, domain, lr, device, input_preproc,
                        output_preproc, fixed_masks = dropout_masks, 
                        ensemble_size = ensemble_size)

        train_loss = nflows_ensemble_model.train_1d(epochs, train_data, output_postproc)

        inps_normed = torch.tensor(datarange, dtype=torch.float32).reshape(-1,1)
        inps_normed = input_preproc(inps_normed, nflows_ensemble_model.stats_inputs)
        inps = torch.tensor(train_data[0], dtype = torch.float32)
        inps = nflows_ensemble_model.input_preproc(inps, nflows_ensemble_model.stats_inputs)
        fit_data_nflows_ensemble = nflows_ensemble_model.sample(1, inps.reshape(-1,1), ensemble_size=ensemble_size)
        fit_data_nflows_ensemble = fit_data_nflows_ensemble[0].detach().cpu()
        fit_data_nflows_ensemble = output_postproc(fit_data_nflows_ensemble.reshape(-1,1), nflows_ensemble_model.stats_outputs)
        sample_size = 20000
        uncertainty_estimates_nflows_ensemble, time_taken = estimate_uncertainty(inps_normed, 
            'all', nflows_ensemble_model, ensemble_size, numb_samps= sample_size, 
            nflows=True, epi_estimator= eue, estimator_types=estimator_types)
        uncertainty_estimates.append(uncertainty_estimates_nflows_ensemble)
        samps_model.append(fit_data_nflows_ensemble)

    if model_type == 'mc_drop':
    
        num_layers = 5
        hids = 400
        output_dim = 1
        context_dim = 1
        device = 'cpu'
        lr = 0.0005
        epochs = 5000
        ensemble_size = 5 

        mc_drop_model = mc_drop(num_layers, hids, output_dim, context_dim,
                        lr, device, input_preproc, output_preproc)
        train_loss = mc_drop_model.train_1d(epochs, train_data, output_postproc)
        inps = torch.tensor(train_data[0], dtype = torch.float32)
        inps = mc_drop_model.input_preproc(inps, mc_drop_model.stats_inputs)
        fit_data_mc_drop = mc_drop_model.sample(1, inps.reshape(-1,1), ensemble_size=ensemble_size)
        fit_data_mc_drop = fit_data_mc_drop[0].detach().cpu()
        fit_data_mc_drop = output_postproc(fit_data_mc_drop.reshape(-1,1), mc_drop_model.stats_outputs)
        sample_size = 5000
        inps = torch.tensor(datarange, dtype=torch.float32).reshape(-1,1)
        inps_normed = input_preproc(inps, mc_drop_model.stats_inputs)
        uncertainty_estimates_mc_drop, time_taken = estimate_uncertainty(inps_normed,                
            'all', mc_drop_model, 20, numb_samps= sample_size,
            epi_estimator= eue, estimator_types=estimator_types)
        uncertainty_estimates.append(uncertainty_estimates_mc_drop)
        samps_model.append(fit_data_mc_drop)

    if model_type == 'pens':
        num_layers = 1 
        hids = 50
        output_dim = 1
        context_dim = 1
        ensemble_size = 5
        device = 'cpu'
        lr = 0.0005
        epochs = 10000
        if just_legend:
            epochs = 200
        dropout_masks = True

        pens_model = pens(num_layers, hids, output_dim, context_dim,
                        lr, device, input_preproc, output_preproc,
                        fixed_masks = dropout_masks, ensemble_size = ensemble_size)
        train_loss = pens_model.train_1d(epochs, train_data, output_postproc)
        inps = torch.tensor(train_data[0], dtype = torch.float32)
        inps = pens_model.input_preproc(inps, pens_model.stats_inputs)
        fit_data_pens = pens_model.sample(1, inps.reshape(-1,1), ensemble_size=ensemble_size)
        fit_data_pens = fit_data_pens[0].detach().cpu()
        fit_data_pens = output_postproc(fit_data_pens.reshape(-1,1), 
            pens_model.stats_outputs)
        sample_size = 20000
        inps = torch.tensor(datarange, dtype=torch.float32).reshape(-1,1)
        inps_normed = input_preproc(inps, pens_model.stats_inputs)
        uncertainty_estimates_pens, time_taken = estimate_uncertainty(inps_normed,
            'all', pens_model, ensemble_size, numb_samps= sample_size,
            epi_estimator= eue, estimator_types=estimator_types)
        uncertainty_estimates.append(uncertainty_estimates_pens)
        samps_model.append(fit_data_pens)


train_x = [i[0] for i in train_data_all]
train_x = np.hstack(train_x)
train_y = [i[1] for i in train_data_all]
train_y = np.hstack(train_y)
np.save(os.path.join(data_path, f'{env}_train_x.npy'), train_x)
np.save(os.path.join(data_path, f'{env}_train_y.npy'), train_y)
if env =='hetero':
    datarange = np.linspace(-4.5, 4.5,numb_uncertainty_estimates)
    kp = (train_x<4.5)&(train_x>-4.5)
    train_x = train_x[kp]
    train_y = train_y[kp]
else:
    datarange = np.linspace(0, 3.5, numb_uncertainty_estimates)
    kp = (train_x<3.5)
    train_x = train_x[kp]
    train_y = train_y[kp]

np.save(os.path.join(data_path, f'{env}_samps_{model_type}.npy'), np.vstack(samps_model).squeeze())
samps_model = np.vstack(samps_model).squeeze()

uncertainty_mib = [i['sample_bald'] for i in uncertainty_estimates]
uncertainty_mib = np.stack(uncertainty_mib)
np.save(os.path.join(data_path, f'{env}_bald_{model_type}.npy'), uncertainty_mib)

uncertainty_bhatt = [i['bhatt_exp'] for i in uncertainty_estimates]
uncertainty_bhatt = np.stack(uncertainty_bhatt)
np.save(os.path.join(data_path, f'{env}_bhatt_{model_type}.npy'), uncertainty_bhatt)

uncertainty_kl = [i['kl_exp'] for i in uncertainty_estimates]
uncertainty_kl = np.stack(uncertainty_kl)
np.save(os.path.join(data_path, f'{env}_kl_{model_type}.npy'), uncertainty_kl)

uncertainty_alea = [i['alea_unc'] for i in uncertainty_estimates]
uncertainty_alea = np.stack(uncertainty_alea)
np.save(os.path.join(data_path, f'{env}_alea_unc_{model_type}.npy'), uncertainty_alea)

if just_legend:
    title_size = 10
    line_width = 4
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), sharey=True, sharex=True)
    ax2 = ax.twinx()
    ax.scatter(train_x, samps_model[kp], s=2, c=color_hexes[4])
    ax2.plot(datarange, uncertainty_mib.mean(0),
        c=color_hexes[0], linewidth=line_width, label = 'MC')
    ax2.plot(datarange, uncertainty_bhatt.mean(0), '--',
        c=color_hexes[1], linewidth=line_width, label='Bhatt')
    ax2.plot(datarange, uncertainty_kl.mean(0), ':',
        c=color_hexes[2], linewidth=line_width, label='KL')
    if model_type == 'nflows_base':
        title = 'Nflows Base'
    elif model_type == 'pens':
        title = 'PNEs'
    elif model_type == 'mc_drop':
        title = 'MC Droput'
    ax.set_title(title, fontdict={'fontsize':title_size})
    ax.set_xticks([])
    ax.set_yticks([])
    ax2.set_yticks([])
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    leg = fig.legend(lines, labels, loc='lower center', ncol=3, fontsize=title_size, bbox_to_anchor=(.5, -.5))
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor('black')
    export_legend(leg)
else:
    title_size = 15
    line_width = 4
    tick_size = 14
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, sharex=True)
    ax2 = ax[0].twinx()
    sns.histplot(data = train_x, color = color_hexes[5],
                 stat = 'probability', ax=ax2, bins=100, alpha=0.5,
                 label='density')
    ax[0].scatter(train_x, train_y, s=2, c=color_hexes[4], label='groundtruth')
    ax2.set(ylabel='')
    ax[0].set_title('Train Data', fontdict={'fontsize':title_size})
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax2.set_yticks([])
    ax2 = ax[1].twinx()
    ax[1].scatter(train_x, samps_model[kp], s=2, c=color_hexes[4])
    ax2.plot(datarange, ((uncertainty_mib.mean(0)-uncertainty_mib.mean(0).min())/
        (uncertainty_mib.mean(0).max()-uncertainty_mib.mean(0).min())),
        c=color_hexes[0], linewidth=line_width, label = 'MC')
    ax2.plot(datarange, ((uncertainty_bhatt.mean(0)-uncertainty_bhatt.mean(0).min())/
        (uncertainty_bhatt.mean(0).max()-uncertainty_bhatt.mean(0).min())), '--',
        c=color_hexes[1], linewidth=line_width, label='Bhatt')
    ax2.plot(datarange, ((uncertainty_kl.mean(0)-uncertainty_kl.mean(0).min())/
        (uncertainty_kl.mean(0).max()-uncertainty_kl.mean(0).min())), ':',
        c=color_hexes[2], linewidth=line_width, label='KL')
    if model_type == 'nflows_base':
        title = 'Nflows Base'
    elif model_type == 'pens':
        title = 'PNEs'
    elif model_type == 'mc_drop':
        title = 'MC Droput'
    ax[1].set_title(title, fontdict={'fontsize':title_size})
    ax[1].set_xticks([])
    fig.tight_layout()
    png_name = f'{env}'
    if model_type == 'pens':
        png_name += '_pnes'
    if model_type == 'mc_drop':
        png_name += '_mc_dropout'
    if model_type == 'nflows_base':
        png_name += '_nflows_base'
    png_name += '.png'
    plt.savefig(os.path.join(graph_path, png_name), dpi=300)
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, sharex=True)
    ax2 = ax[0].twinx()
    sns.histplot(data = train_x, color = color_hexes[5],
                 stat = 'probability', ax=ax2, bins=100, alpha=0.5,
                 label='density')
    ax[0].scatter(train_x, train_y, s=2, c=color_hexes[4], label='groundtruth')
    ax2.set(ylabel='')
    ax[0].set_title('Train Data', fontdict={'fontsize':title_size})
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax2.set_yticks([])
    ax2 = ax[1].twinx()
    ax[1].scatter(train_x, samps_model[kp], s=2, c=color_hexes[4])
    ax2.plot(datarange, uncertainty_mib.mean(0),
        c=color_hexes[0], linewidth=line_width, label = 'Monte Carlo')
    ax2.plot(datarange, uncertainty_bhatt.mean(0), '--',
        c=color_hexes[1], linewidth=line_width, label='Bhatt')
    ax2.plot(datarange, uncertainty_kl.mean(0), ':',
        c=color_hexes[2], linewidth=line_width, label='KL')
    if model_type == 'nflows_base':
        title = 'Nflows Base'
    elif model_type == 'pens':
        title = 'PNEs'
    elif model_type == 'mc_drop':
        title = 'MC Droput'
    ax[1].set_title(title, fontdict={'fontsize':title_size})
    ax[1].set_xticks([])
    fig.tight_layout()
    png_name = f'{env}'
    if model_type == 'pens':
        png_name += '_pnes'
    if model_type == 'mc_drop':
        png_name += '_mc_dropout'
    if model_type == 'nflows_base':
        png_name += '_nflows_base'
    png_name += '_nomap'
    png_name += '.png'
    plt.savefig(os.path.join(graph_path, png_name), dpi=300)
    plt.close()
