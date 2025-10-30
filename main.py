import json
import os
import argparse
import sys
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")


from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

from replay_buffer import ReplayMemory, load_mem_uncertain
from analyze_fit import calc_rmse_1d, calc_rmse
from estimate_uncertainty import find_best_points_1d, find_best_points
from utils import (instantiate_model, normalize, un_normalize,
    identity, gen_folder_uncertain, seed_everything)
from envs_1d import (hetero_samp, hetero_samp_unif, hetero_samp_test, bimodal_samp, bimodal_samp_unif,
    hetero_samp_condition, bimodal_samp_condition)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="hetero", help='Environment [bimodal, hetero, WetChicken-v0, Pendulum-v0, HalfCheetah-v2, Hopper-v2]')
    parser.add_argument('--seed', type=int, default=1456, help='random seed (default: 123456)')
    parser.add_argument('--num_layers', default=1, help='total number of flows', type = int)
    parser.add_argument('--hids', type = int, default = 50, help='hidden units in flows')
    parser.add_argument('--lr', default=5e-4, type=float, help='flows learning rate')
    parser.add_argument('--batch_size', default=256, type=int, help='size of training batch size')
    parser.add_argument('--epochs', default=6000, type=int, help='number of epochs for model')
    parser.add_argument('--model', default="nflows_ensemble", type =str, help='Selects the dynamics model [mc_drop, nn_ensemble, nflows_ensemble])')
    parser.add_argument('--ensemble_size', default=5, type = int, help='number of components in uncertainty models')
    parser.add_argument('--epochs_multiplier', type=int, default=100, help='number of printouts')
    parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
    parser.add_argument('--rqs', action="store_true", help='rational quadratic or cubic spline')
    parser.add_argument('--dropout_masks', action="store_true", help='fixed set of dropout masks')
    parser.add_argument('--base_distro', action="store_true", help='ensemble in base distro')
    parser.add_argument('--acquisition_function', type=str, default='sample_bald', help='how to acquire new points')
    parser.add_argument('--test_num_samples', action="store_true", help='test different number of samples for MC on Hopper-v2')
    parser.add_argument('--numb_samps', type=int, default=10, help='numb samps for MC test num_samples')
    parser.add_argument('--save_model', action= 'store_true', help='save model or not')
    parser.add_argument('--bootstrap', action= 'store_true', help='whether or not to bootstrap the data, used for model PE')
    parser.add_argument('--noise_weight', type=float, default=0.2, help='how much noise to add in')
    parser.add_argument('--modes', default=0, type=int, help='number of modes in noise to simulate chaotic dynamics')
    parser.add_argument('--replay_size', type=int, default=1000000, help='size of replay buffer (default: 10000000)')
    parser.add_argument('--data_size', type=int, default=200, help='controls size of the data (negative number use all data)')
    parser.add_argument('--uncertain_nflows', action="store_true", help='uncertainty in nflow layers')
    parser.add_argument('--points_2_add', type=int, default=10, help='how many new points to acquire')
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)
    store_dir = './results'
    save_model_dir = './models'
    branch_folder, child_folder = gen_folder_uncertain(args)
    env_dir = os.path.join(store_dir, branch_folder)
    store_dir = os.path.join(env_dir, child_folder)
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    save_model_dir = os.path.join(save_model_dir, branch_folder)
    save_model_dir = os.path.join(save_model_dir, child_folder)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    results_dir = os.path.join(store_dir, 'results/')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    imgs_dir = os.path.join(store_dir, 'epoch_imgs/')
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    with open(os.path.join(save_model_dir, 'date_ran.txt'), mode='a') as f:
        f.write(f'Date: \n{time.strftime("%Y-%m-%d_%H_%M_%S")}')
    epoch_files = os.listdir(imgs_dir)
    for f in epoch_files:
        path = os.path.join(imgs_dir, f)
        os.remove(path)
    results_files = os.listdir(results_dir)
    for f in results_files:
        path = os.path.join(results_dir, f)
        os.remove(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_dims = 1
    output_dim = 1
    numb_samps = 1000
    ensemble = True
    nflows = False
    one_d =  True
    num_points_2_add = args.points_2_add
    if args.model == 'nflows_ensemble':
        nflows = True
    if args.env == 'bimodal':
        train_data = bimodal_samp(100)
        test_data = bimodal_samp_unif(20000)
        oracle_data = bimodal_samp(100000)
        gt_sampler = bimodal_samp_condition
        train_set_size = [len(train_data[0])]
    elif args.env == 'hetero':
        train_data = hetero_samp(100)
        test_data = hetero_samp_test(20000)
        oracle_data = hetero_samp(100000)
        gt_sampler = hetero_samp_condition
        train_set_size = [len(train_data[0])]
    else:
        one_d =  False
        memory = ReplayMemory(args.replay_size, args.batch_size, bootstrap = args.bootstrap,
            ensemble_size = args.ensemble_size, shuffle = True)
        load_mem_uncertain(args, memory, env_dir)
        test_memory = ReplayMemory(args.replay_size, 1028, bootstrap = args.bootstrap,
                ensemble_size = args.ensemble_size, shuffle = False)
        load_mem_uncertain(args, test_memory, env_dir, dataset='test')
        if args.data_size > 0:
            memory.reduce_buffer(args.data_size)
        test_memory.reduce_buffer(2000)
        oracle_memory = ReplayMemory(args.replay_size, args.batch_size,
            bootstrap = args.bootstrap, ensemble_size = args.ensemble_size,
            shuffle = False)
        load_mem_uncertain(args, oracle_memory, env_dir, dataset='oracle')
        oracle_memory.remove_portion(memory.buffer)
        # Correction for contact forces which were broken in Mujoco v2
        if args.env=='Pendulum-v0':
            state_dim = 3 
            action_dim = 1 
        elif args.env=='Ant-v2':
            state_dim = 27
            action_dim = 8 
        elif args.env=='Hopper-v2':
            state_dim = 11
            action_dim = 3 
        elif args.env=='Humanoid-v2':
            state_dim = 257
            action_dim = 17 
        context_dims = action_dim+state_dim
        action_dim_seq = action_dim
        args.output_dim = state_dim
        output_dim = state_dim
        args.context_dim = context_dims
        numb_samps = 5000
        if args.model == 'nflows_ensemble':    
            nflows = True
        if args.acquisition_function == 'batchbald':
            numb_samps = 1000
        if args.test_num_samples:
            numb_samps = args.numb_samps
        if args.test_num_samples:
            numb_samps = args.numb_samps
        train_set_size = [len(memory.buffer)]
    with open(os.path.join(store_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(save_model_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    model = instantiate_model(args, output_dim, context_dims, device, normalize,
        normalize)
    test_losses = []
    rmses = []
    time_estimates = []
    for i in range(args.epochs_multiplier):
        start_time = time.time()
        if one_d:
            train_loss = model.train_1d(args.epochs, train_data, un_normalize)
            model.detach_model()
            test_loss = model.loss_1d(test_data)
        else:
            train_loss = model.train(args.epochs, memory)
            model.detach_model()
            test_loss = model.loss(test_memory)
        epoch_suffix = 'epoch_'+str(((i+1)))
        if one_d:
            idxs = np.random.choice(oracle_data[0].shape[0], 1000, replace=False)
            samp_oracle = (oracle_data[0][idxs], oracle_data[1][idxs])
        else:
            if args.env == 'Humanoid-v2':
                samp_oracle = oracle_memory.sample(125)
                if args.acquisition_function in ['batchbald', 'bait', 'badge']:
                    samp_oracle = oracle_memory.sample(1000)
            else:
                samp_oracle = oracle_memory.sample(1000)
            if args.acquisition_function not in ['sample_bald', 'batchbald', 'bait', 'badge']:
                samp_oracle = oracle_memory.sample(10000)
        if args.acquisition_function != 'random':
            if one_d:
                x_train = train_data[0].reshape(-1,1)
                y_train = train_data[1].reshape(-1,1)
                x_train = model.input_preproc(x_train, model.stats_inputs)
                y_train = model.input_preproc(y_train, model.stats_outputs)
                points_2_add, time_taken = find_best_points_1d(samp_oracle,
                    int(numb_samps/args.ensemble_size), model, normalize,
                    args.ensemble_size, device, acquisition_criteria = args.acquisition_function,
                    nflows=nflows, x_train=x_train, y_train=y_train)
            else:
                states, actions, _, next_states, _, _, _ = map(np.stack, zip(*memory.buffer))
                states = torch.tensor(states, dtype = torch.float32).to(model.device)
                actions = torch.tensor(actions, dtype = torch.float32).to(model.device)
                next_states = torch.tensor(next_states, dtype = torch.float32).to(model.device)
                inps = torch.hstack([states, actions])
                outs = next_states
                inps = model.input_preproc(inps, model.stats_inputs)
                outs = model.output_preproc(outs, model.stats_outputs)
                points_2_add, time_taken = find_best_points(samp_oracle, numb_samps,
                    model, normalize, args.ensemble_size,
                    device, acquisition_criteria = args.acquisition_function,
                    nflows=nflows, numb_points_2_add = num_points_2_add,
                    x_train=inps, y_train=outs)
        else:
            if one_d:
                idxs = np.random.choice(samp_oracle[0].shape[0], 10, replace=False)
                points_2_add = (samp_oracle[0][idxs], samp_oracle[1][idxs])
            else:
                rand_samp = oracle_memory.sample(num_points_2_add)
                points_2_add = [(rand_samp[0][i], rand_samp[1][i],
                    rand_samp[2][i], rand_samp[3][i], rand_samp[4][i],
                    rand_samp[5][i], rand_samp[6][i]) for i in range(num_points_2_add)] 
            time_taken=0
            torch.cuda.empty_cache()
        time_estimates.append(time_taken)
        np.save(os.path.join(env_dir, 'time_estimates_'+args.acquisition_function),np.array(time_estimates))
        if one_d:
            train_data = (np.concatenate([train_data[0], points_2_add[0]]),
                np.concatenate([train_data[1], points_2_add[1]]))
            rmse = calc_rmse_1d(test_data, normalize, un_normalize,
                model, ensemble_size = args.ensemble_size, device = device)
        else:
            memory.add_to_buffer(points_2_add)
            oracle_memory.remove_portion(points_2_add)
            rmse = calc_rmse(test_memory, normalize, un_normalize,
                model, ensemble_size = args.ensemble_size, device = device)

        train_losses = train_loss
        test_losses += [test_loss]
        rmses.append(rmse)
        mean_dyna_loss = torch.tensor(train_loss).mean()
        if one_d:
            train_set_size.append(len(train_data[0]))
        else:
            train_set_size.append(len(memory.buffer))
        end_time = time.time()
        train_time = str(timedelta(seconds=(end_time-start_time)))
        performance_string = f'Epoch: {(i+1)}, '\
                             f'Train Loss: {mean_dyna_loss:.2f}, '\
                             f'test Loss: {test_loss:.2f}, '\
                             f'Train Time: {train_time}'
        print(performance_string)
        if one_d:
            print(f'RMSE Test: {rmse:.2f}, '\
                f' Train Set Size: {len(train_data[0])-num_points_2_add}')
        else:
            print(f'Last Train Loss: {train_loss[-1]:.2f}, RMSE Test: {rmse}, '\
                f' Train Set Size: {len(memory.buffer)-num_points_2_add}')
        np.save(os.path.join(results_dir, ('train_loss_array')), np.array(train_losses))
        np.save(os.path.join(results_dir, ('test_loss_array')), np.array(test_losses))
        np.save(os.path.join(results_dir, ('rmse_array')), np.array(rmses))
        print('Saving Model')
        model_path = os.path.join(save_model_dir,('model.pt'))
        model.save_model(model_path)
        model = instantiate_model(args, output_dim, context_dims, device, normalize,
            normalize)
        print("-----------------------------------------------")
