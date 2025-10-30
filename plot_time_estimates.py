import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', family='Times New Roman')
color_hexes = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_time_estimates(bhatt, kl, mc, bhatt_ensembles, kl_ensembles, store_dir):
    fig, ax = plt.subplots(1, 2, figsize=(4.5, 2), sharey=True)
    tick_size = 8 
    title_size = 10
    marker_size = 4 
    linew = 2 
    x = [1, 3, 11, 27, 257]
    envs = ['bimodal', 'Pendulum-v0', 'Hopper-v2', 'Ant-v2', 'Humanoid-v2']
    bhatt_means = []
    kl_means = []
    mc_means = []
    bhatt_stds = []
    kl_stds = []
    mc_stds = []
    for i, env in enumerate(envs):
        bhatt_estimates = bhatt[env]
        kl_estimates = kl[env]
        mc_estimates = mc[env]
        bhatt_means.append(bhatt_estimates.mean())
        kl_means.append(kl_estimates.mean())
        mc_means.append(mc_estimates.mean())
        bhatt_stds.append(bhatt_estimates.std())
        kl_stds.append(kl_estimates.std())
        mc_stds.append(mc_estimates.std())
    ax[0].plot(x, mc_means, '-o', label='MC', c=color_hexes[3], alpha=0.7, markersize=marker_size, 
        linewidth=linew)
    ax[0].plot(x, bhatt_means, '-o', label='Bhatt', c=color_hexes[0], alpha=0.7, 
        markersize=marker_size, linewidth=linew)
    ax[0].plot(x, kl_means, '-o', label='KL', c=color_hexes[1], alpha=0.7, 
        markersize=marker_size, linewidth=linew)
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_xlabel("Number of Dimensions", fontsize=title_size, labelpad=1)
    ax[0].set_ylabel("Seconds",  fontsize=title_size, labelpad=1)
    ax[0].tick_params(axis='x', labelsize=tick_size)
    ax[0].tick_params(axis='y', labelsize=tick_size)
    x = bhatt_ensembles.keys()
    bhatt_means = []
    kl_means = []
    for numb in x:
        bhatt_estimates = bhatt_ensembles[numb]
        kl_estimates = kl_ensembles[numb]
        bhatt_means.append(bhatt_estimates.mean())
        kl_means.append(kl_estimates.mean())
        bhatt_stds.append(bhatt_estimates.std())
        kl_stds.append(kl_estimates.std())
    ax[1].plot(x, bhatt_means, '-o', label='Bhatt', c=color_hexes[0], alpha=0.7, 
        markersize=marker_size, linewidth=linew)
    ax[1].plot(x, kl_means, '-o', label='KL', c=color_hexes[1], alpha=0.7, 
        markersize=marker_size, linewidth=linew)
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel("Number of Ensembles", fontsize=title_size, labelpad=1)
    ax[1].tick_params(axis='x', labelsize=tick_size)
    ax[1].tick_params(axis='y', labelsize=tick_size)
    lines = ax[0].get_legend_handles_labels()[0]
    labels = ax[0].get_legend_handles_labels()[1]
    leg = fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(.5, -.3), ncol=4, fontsize="8")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_edgecolor('black')
    plt.savefig(os.path.join(store_dir, 'time_estimates.png'), dpi=400, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    base_dir = './results/'
    envs = ['bimodal','Pendulum-v0', 'Hopper-v2', 'Ant-v2', 'Humanoid-v2']
    all_time_estimates_bhatt = {}
    all_time_estimates_kl = {}
    all_time_estimates_mc = {}
    for env in envs:
        env_dir = os.path.join(base_dir, env)
        time_estimates_bhatt = np.load(os.path.join(env_dir,'time_estimates_bhatt_exp.npy'), 
            allow_pickle=True)
        time_estimates_kl = np.load(os.path.join(env_dir,'time_estimates_kl_exp.npy'),
            allow_pickle=True)
        time_estimates_mc = np.load(os.path.join(env_dir,'time_estimates_mutual_info.npy'),
            allow_pickle=True)
        all_time_estimates_bhatt[env] = time_estimates_bhatt
        all_time_estimates_kl[env] = time_estimates_kl
        all_time_estimates_mc[env] = time_estimates_mc
    number_ensembles = [5, 10, 50, 100]
    time_estimates_bhatt_number_ensembles = {}
    time_estimates_kl_number_ensembles = {}
    env_dir = os.path.join(base_dir, 'Humanoid-v2')
    env_dir = os.path.join(env_dir, 'ensemble_size_comp')
    for numb in number_ensembles:
        time_estimates_bhatt = np.load(os.path.join(env_dir,f'time_estimates_bhatt_exp_{numb}.npy'),
            allow_pickle=True)
        time_estimates_kl = np.load(os.path.join(env_dir,f'time_estimates_kl_exp_{numb}.npy'),
            allow_pickle=True)
        time_estimates_bhatt_number_ensembles[numb] = time_estimates_bhatt
        time_estimates_kl_number_ensembles[numb] = time_estimates_kl

    plot_time_estimates(all_time_estimates_bhatt, all_time_estimates_kl, 
        all_time_estimates_mc, time_estimates_bhatt_number_ensembles, time_estimates_kl_number_ensembles, 
        './graphs')
