import argparse
from datetime import datetime
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('font', family='Times New Roman')
#color_hexes = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', 
#        '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
color_hexes = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_1stdv_multi(x, x_2, x_3, metrics, ylabel, filename, title, store_dir, envs, acq_crit, rmse=True):
    #fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    #tick_size = 12
    #axes_size = 14
    #title_size = 14
    fig, ax = plt.subplots(2, 3, figsize=(6, 3),  gridspec_kw={'hspace': 0.25})
    tick_size = 4
    font_size2 = 6 
    axes_size = font_size2
    title_size = font_size2
    for i, metric in enumerate(envs):
        models = metrics[metric].keys()
        ci = 0 
        domain = x
        row = int(i/3 % 2)
        col = int(i % 3)
        if metric in ['Pendulum-v0', 'Hopper-v2', 'Ant-v2']:
            domain = x_2
        elif metric in ['Humanoid-v2']:
            domain = x_3
        plot_labels = {'sample_bald':'MC', 'bhatt_exp':'Bhatt', 'kl_exp':'KL', 'random':'Random',
            'bmdal_batchbald':'BatchBALD', 'bmdal_badge':'BADGE', 'bmdal_bait':'BAIT', 'bmdal_lcmd':'LCMD'}
        for m in acq_crit:
            clipped =metrics[metric][m].clip(-300,300)
            #clipped = y_models[m]
            mean = np.nanmean(clipped, axis=0)
            sigma = np.nanstd(clipped, axis=0) 
            ax[row, col].plot(domain, mean, label=plot_labels[m], c=color_hexes[ci], lw=1)
            ax[row, col].fill_between(domain, mean+sigma, mean-sigma, 
                alpha=0.5, color=color_hexes[ci])
            ci += 1
        ax[row, col].set_title(metric, fontdict={'fontsize':title_size, 'style': 'italic'}, pad=1.5)
        #ax[row, col].set_title(metric, fontdict={'style': 'italic'}, pad=1.5)
        ax[row, col].tick_params(axis='x', labelsize=tick_size)
        ax[row, col].tick_params(axis='y', labelsize=tick_size)
    if rmse:
        ax[0, 0].set_ylabel('RMSE', fontsize=axes_size)
        ax[1, 0].set_ylabel('RMSE', fontsize=axes_size)
        #ax[0, 0].set_ylabel('RMSE')
        #ax[1, 0].set_ylabel('RMSE')
    else:
        ax[0, 0].set_ylabel('Log Likelihood', fontsize=axes_size)
        ax[1, 0].set_ylabel('Log Likelihood', fontsize=axes_size)
    ax[1, 0].set_xlabel('Training Set Size', fontsize=axes_size)
    ax[1, 1].set_xlabel('Training Set Size', fontsize=axes_size)
    ax[1, 2].set_xlabel('Training Set Size', fontsize=axes_size)
    lines = ax[0,0].get_legend_handles_labels()[0]
    labels = ax[0,0].get_legend_handles_labels()[1]
    leg = fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(.5, -.12), ncol=4, fontsize=f"{font_size2}", borderpad=0.65)
    for line in leg.get_lines():
        line.set_linewidth(6.0)
    leg.get_frame().set_linewidth(1)
    leg.get_frame().set_edgecolor('black')
    plt.savefig(os.path.join(store_dir, filename), dpi=500, bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    base_dir = './results/'
    #models_to_graph = 'nflows_ensemble_fixedmasks'
    #title_multi = 'Nflows Ensemble'
    #model_name = 'nflows_ensemble_row'
    models_to_graph = 'nn_ensemble_fixedmasks'
    title_multi = 'NN Ensemble'
    model_name = 'pens_row'
    acq_crit = ['bhatt_exp', 'kl_exp', 'random', 'sample_bald', 'batchbald', 
        'badge', 'bait']
    cutoff = 100
    envs = ['hetero', 'bimodal', 'Pendulum-v0', 'Hopper-v2', 'Ant-v2', 'Humanoid-v2']
    all_rmses = {}
    all_likelihoods = {}
    for env in envs:
        env_dir = os.path.join(base_dir, env)
        run_dirs = os.listdir(env_dir)
        run_dirs = [os.path.join(env_dir, d) for d in run_dirs 
            if os.path.isdir(os.path.join(env_dir, d))]
        rmses_models = {}
        likelihoods_models = {}
        for crit in acq_crit:
            model_dirs = [d for d in run_dirs 
                if crit in os.path.basename(os.path.normpath(d))]
            model_dirs = [d for d in model_dirs if models_to_graph in d]
            rmses = []
            likelihoods = []
            for d in model_dirs:
                if d.split('_')[-1] == 'seed585':
                    continue
                if models_to_graph not in d:
                    continue
                args_path = os.path.join(d,'commandline_args.txt')
                saved_parser = argparse.ArgumentParser()
                saved_args = saved_parser.parse_args()
                with open(args_path, 'r') as f:
                    saved_args.__dict__ = json.load(f)
                train_len = saved_args.epochs_multiplier
                results_dir = os.path.join(d, 'results')
                try:
                    rmse = np.load(os.path.join(results_dir, 'rmse_array.npy'))
                except:
                    print(f'dir {results_dir} did not work')
                    continue
                test_loss = np.load(os.path.join(results_dir, 'test_loss_array.npy')) 
                likelihood = -test_loss
                rmse_nans = np.where(np.isnan(rmse))
                likelihoods_nans = np.where(np.isnan(likelihood))
                print(d)
                print(datetime.utcfromtimestamp(os.path.getmtime(os.path.join(results_dir, 'rmse_array.npy'))).strftime('%Y-%m-%d %H:%M:%S'))
                print(rmse.shape)
                if len(rmse) >= cutoff:
                    rmses.append(rmse[:cutoff])
                    likelihoods.append(likelihood[:cutoff])
            rmses = np.stack(rmses)
            likelihoods = np.stack(likelihoods)
            rmses_models[crit] = rmses
            likelihoods_models[crit] = likelihoods
        all_rmses[env] = rmses_models
        all_likelihoods[env] = likelihoods_models
    train_size = np.linspace(0,cutoff-1, cutoff)*10+100 
    train_size_multi = np.linspace(0,cutoff-1, cutoff)*10+200 
    plot_1stdv_multi(train_size, train_size_multi, train_size_multi, all_rmses, 
        'RMSE','rmse_'+model_name+'.png', 'training size', 
        './graphs', envs, acq_crit)
    plot_1stdv_multi(train_size, train_size_multi, train_size_multi, all_likelihoods, 
        'Log Likelihood','log_likelihood_'+model_name+'.png', 'training size', 
        './graphs', envs, acq_crit, rmse=False)
