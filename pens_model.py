import os
import argparse
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

from base_model import base_model
from pens_utils import build_pens

class pens(base_model):
    def __init__(self, num_layers, hids, dims, context_dims, 
            lr, device, input_preproc, output_preproc, 
            multihead=False, fixed_masks=True, ensemble_size=5):
        self.dims = dims
        self.model = build_pens(hids=hids, dims=dims, 
            context_dims=context_dims, device = device, multihead=multihead,
            fixed_masks=fixed_masks, ensemble_size=ensemble_size,
            num_layers=num_layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.output_preproc = output_preproc
        self.input_preproc = input_preproc
        self.ensemble_size = ensemble_size
        self.fixed_masks = fixed_masks
    
    def train_1d(self, epochs, data, un_normalize):
        train_losses = []
        self.set_stats_1d(data)
        for epoch in range(epochs):
            running_train_loss = 0
            inp = data[0].reshape(-1,1)
            out = data[1].reshape(-1,1)
            inps = torch.tensor(inp, dtype = torch.float32).to(self.device)
            outs = torch.tensor(out, dtype = torch.float32).to(self.device)
            inps = self.input_preproc(inps, self.stats_inputs)
            outs = self.output_preproc(outs, self.stats_outputs)
            self.optimizer.zero_grad()
            mu, sig, mask_prob = self.model(inps)
            loss = -self.model.loss_val(mu, sig, outs)
            loss.backward()
            self.optimizer.step()
            train_loss = loss.cpu().detach()
            train_losses.append(train_loss)
            if (epoch + 1) % int(epochs/5) == 0:
                progress = (epoch / epochs) * 100
                rounded_progress = round(progress / 20) * 20
                print(f'training {rounded_progress}% complete loss: {train_loss.item()}')
        return train_losses
    
    def loss_1d(self, data):
        with torch.no_grad():
            inp = data[0].reshape(-1,1)
            out = data[1].reshape(-1,1)
            inps = torch.tensor(inp, dtype = torch.float32).to(self.device)
            outs = torch.tensor(out, dtype = torch.float32).to(self.device)
            inps = self.input_preproc(inps, self.stats_inputs)
            outs = self.output_preproc(outs, self.stats_outputs)
            loss = 0
            for ei in range(self.ensemble_size):
                kwargs = {'rand_mask': False, 'mask_index': ei}
                mu, sig, mask_prob = self.model.forward(inps, **kwargs)
                comp_log_prob = self.model.log_prob(mu, sig, outs)
                loss += (torch.exp(comp_log_prob)*1/self.ensemble_size)
            loss = torch.log(loss)
            loss[loss.isinf()] = -250
            loss = -loss.mean().cpu().detach()
        return loss 

    def train(self, epochs, data_loader):
        train_losses = []
        self.set_stats(data_loader)
        for epoch in range(epochs):
            running_train_loss = 0
            total_inputs = 0
            for data in data_loader:
                states = data[0]
                actions = data[1]
                next_states = data[3]
                states = torch.tensor(states, dtype = torch.float32).to(self.device)
                actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
                next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
                inps = torch.hstack([states, actions])
                outs = next_states
                inp_stats = self.stats_inputs + [states]
                out_stats = self.stats_outputs + [states]
                inps = self.input_preproc(inps, inp_stats)
                outs = self.output_preproc(outs, out_stats)
                self.optimizer.zero_grad()
                mu, sig, mask_prob = self.model(inps)
                loss = -self.model.loss_val(mu, sig, outs)
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.cpu().detach()*states.shape[0]
                total_inputs += states.shape[0]
                if (epoch + 1) % int(epochs/5) == 0:
                    progress = (epoch / epochs) * 100
                    rounded_progress = round(progress / 20) * 20
                    print(f'training {rounded_progress}% complete loss: {train_loss.item()}')
            running_train_loss = running_train_loss/total_inputs
            train_losses.append(running_train_loss)
        return train_losses
    
    def loss(self, data_loader):
        running_loss = 0
        total_inputs = 0
        with torch.no_grad():
            for data in data_loader:
                states = data[0]
                actions = data[1]
                next_states = data[3]
                states = torch.tensor(states, dtype = torch.float32).to(self.device)
                actions = torch.tensor(actions, dtype = torch.float32).to(self.device)
                next_states = torch.tensor(next_states, dtype = torch.float32).to(self.device)
                inps = torch.hstack([states, actions])
                outs = next_states
                inp_stats = self.stats_inputs + [states]
                out_stats = self.stats_outputs + [states]
                inps = self.input_preproc(inps, inp_stats)
                outs = self.output_preproc(outs, out_stats)
                loss = 0
                for ei in range(self.ensemble_size):
                    kwargs = {'rand_mask': False, 'mask_index': ei}
                    mu, sig, mask_prob = self.model.forward(inps, **kwargs)
                    comp_log_prob = self.model.log_prob(mu, sig, outs)
                    loss += (torch.exp(comp_log_prob)*1/self.ensemble_size)
                loss = torch.log(loss)
                loss[loss.isinf()] = -250
                running_loss += -loss.sum().cpu().detach()
                total_inputs += states.shape[0]
        running_loss = running_loss/total_inputs
        return running_loss 

    def detach_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def attach_model(self):
        for p in self.model.parameters():
            p.requires_grad = True
    
    def attach_last_layer(self):
        self.model.log_std_linear.weight.requires_grad = True
        self.model.log_std_linear.bias.requires_grad = True
        self.model.mean_linear.weight.requires_grad = True
        self.model.mean_linear.bias.requires_grad = True

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        if self.fixed_masks:
            mask_path = path[:-3]+'_masks'+path[-3:]
            torch.save(self.model.masks, mask_path)
        self.save_constants(path)
    
    def load_model(self, path):
        import pdb; pdb.set_trace()
    
    def sample(self, numb_samps, context, kwargs={},
            ensemble = True, ensemble_size = 10): 
        output_hat, mask_prob, mu, sig = (
            self.model.sample(numb_samps, context = context, kwargs=kwargs))
        return output_hat, mask_prob, mu, sig
    
    def sample_and_log_prob(self, numb_samps, context, kwargs={},
            ensemble = True, ensemble_size = 10):
        samp, log_prob, mask_prob, mu, sig = (
            self.model.sample_and_log_prob(numb_samps, context = context, kwargs=kwargs))
        mu = mu.repeat_interleave(numb_samps, dim=0)
        sig = sig.repeat_interleave(numb_samps, dim=0)
        return samp, log_prob, mu, sig
    
    
    def grad_last_layer(self, x, num_samps=1, bait=False):
        self.attach_last_layer()
        target_hyp, mask_prob, pred_mu, pred_sig = self.sample(num_samps, context=x)
        #target_hyp = target_hyp.squeeze(1)
        all_grads = []
        for j in range(num_samps):
            all_samp_grads = []
            for i in range(x.shape[0]): 
                self.optimizer.zero_grad()
                mu_i = pred_mu[i,:].repeat(1, 1)
                sig_i = pred_sig[i,:].repeat(1, 1)
                loss = -self.model.loss_val(mu_i, sig_i, target_hyp[i, j:j+1,:])
                loss.backward(retain_graph=True)
                last_layer_gradients_mean = torch.cat([self.model.mean_linear.weight.grad.detach().cpu().clone().reshape(-1),
                    self.model.mean_linear.bias.grad.detach().cpu().clone().reshape(-1)])
                last_layer_gradients_std = torch.cat([self.model.log_std_linear.weight.grad.detach().cpu().clone().reshape(-1),
                    self.model.log_std_linear.bias.grad.detach().cpu().clone().reshape(-1)])
                last_layer_gradients = torch.cat([last_layer_gradients_mean, last_layer_gradients_std], axis=0)
                all_samp_grads.append(last_layer_gradients)
            all_grads.append(torch.stack(all_samp_grads))
        self.optimizer.zero_grad()
        self.detach_model()
        all_grads = torch.stack(all_grads).permute(1,0,2)
        all_grads = all_grads.squeeze()
        return all_grads 
