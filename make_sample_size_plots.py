import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

color_hexes = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

mpl.rc('font',family='Times New Roman')

def plot_numssamples_estimates(rmses, llhood, store_dir, rmse=True, nflows=True):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    tick_size = 12
    title_size = 18
    marker_size = 15
    linew = 5
    x = [5, 10, 50, 100, 500, 1000, 5000, 10000, 25000, 50000]
    keys = [1, 2, 10, 20, 100, 200, 1000, 2000, 5000, 10000]
    if nflows:
        peak_perf_rmse = 0.29
        peak_perf_llhood = 18.96
    else:
        peak_perf_rmse = 2.12 
        peak_perf_llhood = 19.47
    if rmse:
        y_rmse_mean = np.array([rmses[i][:,99].mean() for i in keys])
        y_rmse_std = np.array([rmses[i][:,99].std() for i in keys]) 
        ax.plot(x, y_rmse_mean, c=color_hexes[0], label='MC')
        ax.set_xscale('log')
        ax.set_xlabel("Number of Samples", fontsize=title_size)
        ax.set_ylabel("RMSE",  fontsize=title_size)
        ax.axhline(peak_perf_rmse, linestyle='--', c='black', label='PaiDEs Performance')
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        if nflows:
            filename = os.path.join(store_dir, 'number_samples_plot_rmse.png')
        else:
            filename = os.path.join(store_dir, 'number_samples_plot_rmse_pnes.png')
    else:
        y_llhood_mean = np.array([llhood[i][:,99].mean() for i in keys])
        y_llhood_std = np.array([llhood[i][:,99].std() for i in keys]) 
        ax.plot(x, y_llhood_mean, c=color_hexes[0], label='MC')
        ax.set_xscale('log')
        ax.set_xlabel("Number of Samples", fontsize=title_size)
        ax.set_ylabel("Log Likelihood",  fontsize=title_size)
        ax.axhline(peak_perf_llhood, linestyle='--', c='black', label='PaiDEs Performance')
        ax.tick_params(axis='x', labelsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        if nflows:
            filename = os.path.join(store_dir, 'number_samples_plot_llhood.png')
        else:
            filename = os.path.join(store_dir, 'number_samples_plot_llhood_pnes.png')
    lines = ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1]
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(.5, -.2), ncol=4, fontsize="14")


    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    base_dir_samp = './results/Hopper-v2_test_numsamples'
    base_dir_full = './results/Hopper-v2_test_aquisition'
    rmse=False
    nflows=True
    all_rmse = {10000:[], 5000:[], 2000:[], 1000:[], 200:[], 100: [], 20:[], 10:[], 2:[], 1:[]}
    all_llhood = {10000:[], 5000:[], 2000:[], 1000:[], 200:[], 100: [], 20:[], 10:[], 2:[], 1:[]}
    runs = os.listdir(base_dir_samp)
    if nflows:
        runs = [i for i in runs if 'nflows_ensemble' in i]
    else:
        runs = [i for i in runs if 'nn_ensemble' in i]
    for i in tqdm(runs):
        run_dir = os.path.join(base_dir_samp, i)
        result_dir = os.path.join(run_dir, 'results')
        rmses = np.load(os.path.join(result_dir, 'rmse_array.npy'))
        llhood = np.load(os.path.join(result_dir, 'test_loss_array.npy'))
        dict_key = int(i.split('_')[-2]) 
        all_rmse[dict_key].append(rmses)
        all_llhood[dict_key].append(-1*llhood)
    all_rmse[100] = [i for i in all_rmse[100] if i.shape[0]>99]
    all_llhood[100] = [i for i in all_llhood[100] if i.shape[0]>99]
    runs_full = os.listdir(base_dir_full)
    if nflows:
        runs_full = [i for i in runs_full if 'nflows_ensemble_fixedmasks_mutual_info' in i]
    else:
        runs_full = [i for i in runs_full if 'nn_ensemble_fixedmasks_mutual_info' in i]
    for i in tqdm(runs_full):
        run_dir = os.path.join(base_dir_full, i)
        result_dir = os.path.join(run_dir, 'results')
        rmses = np.load(os.path.join(result_dir, 'rmse_array.npy'))
        llhood = np.load(os.path.join(result_dir, 'test_loss_array.npy'))
        all_rmse[5000].append(rmses)
        all_llhood[5000].append(-1*llhood)
    all_rmse = {k:np.stack(v) for (k,v) in all_rmse.items()}
    all_llhood = {k:np.stack(v) for (k,v) in all_llhood.items()}
    plot_numssamples_estimates(all_rmse, all_llhood, './graphs', rmse=rmse, nflows=nflows)
