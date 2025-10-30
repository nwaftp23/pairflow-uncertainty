import argparse
import os
import json

import numpy as np
import matplotlib.pyplot as plt

color_hexes = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', 
        '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']


def plot_time_estimates_number_ensembles(bhatt, kl, store_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    tick_size = 12
    title_size = 18
    x = bhatt.keys()
    bhatt_means = []
    kl_means = []
    bhatt_stds = []
    kl_stds = []
    for numb in x:
        bhatt_estimates = bhatt[numb]
        kl_estimates = kl[numb]
        bhatt_means.append(bhatt_estimates.mean())
        kl_means.append(kl_estimates.mean())
        bhatt_stds.append(bhatt_estimates.std())
        kl_stds.append(kl_estimates.std())
    ax.scatter(x, bhatt_means, label='Bhatt', c=color_hexes[1], alpha=0.7)
    ax.scatter(x, kl_means, label='KL', c=color_hexes[2], alpha=0.7)
    ax.legend(loc='center right')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Number of Ensembles", fontsize=title_size)
    ax.set_ylabel("Seconds",  fontsize=title_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    fig.tight_layout()
    plt.savefig(os.path.join(store_dir, 'time_estimates_number_ensembles.png'), dpi=400)
    plt.close()


if __name__ == '__main__':
    base_dir = './results'
    models_to_graph = 'nflows_ensemble_fixedmasks'
    title_multi = 'Nflows Ensemble'
    model_name = 'nflows_ensemble_row'
    #models_to_graph = 'nn_ensemble_fixedmasks'
    #title_multi = 'NN Ensemble'
    #model_name = 'pens_row'
    number_ensembles = [5, 10, 50, 100]
    all_time_estimates_bhatt = {}
    all_time_estimates_kl = {}
    env_dir = os.path.join(base_dir, 'Humanoid-v2_test_aquisition')
    for numb in number_ensembles:   
        time_estimates_bhatt = np.load(os.path.join(env_dir,f'time_estimates_bhatt_exp_{numb}.npy'), 
            allow_pickle=True)
        time_estimates_kl = np.load(os.path.join(env_dir,f'time_estimates_kl_exp_{numb}.npy'),
            allow_pickle=True)
        all_time_estimates_bhatt[numb] = time_estimates_bhatt
        all_time_estimates_kl[numb] = time_estimates_kl
    plot_time_estimates_number_ensembles(all_time_estimates_bhatt, all_time_estimates_kl, 
        './graphs')
