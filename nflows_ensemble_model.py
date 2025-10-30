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
from nflows_utils import build_nflows_ensemble

class nflows_ensemble(base_model):
    def __init__(self, num_layers, hids, dims, context_dims, 
            bins, tail, lr, device, input_preproc, output_preproc, 
            rqs =True, multihead=False, fixed_masks=False, ensemble_size=15):
        base_ensemble = True
        if ensemble_size == 1:
            base_ensemble == False
        self.model = build_nflows_ensemble(num_layers=num_layers, hids=hids, 
                dims=dims, context_dims=context_dims, batch_norm=False, 
                activation=torch.nn.functional.relu, bins = bins, tail=tail, 
                device = device, rqs =rqs, base=base_ensemble, flows=False, 
                multihead=multihead, fixed_masks=fixed_masks, 
                ensemble_size=ensemble_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.output_preproc = output_preproc
        self.input_preproc = input_preproc
        self.ensemble_size = ensemble_size
        self.fixed_masks = fixed_masks
        self.output_ensemble=False

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
            loss = -self.model.log_prob(outs, context=inps).mean()
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
                comp_log_prob = self.model.log_prob(outs, context=inps, kwargs=kwargs)
                loss += (torch.exp(comp_log_prob)*1/self.ensemble_size)
            loss = torch.log(loss)
            loss[loss.isinf()] = -150
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
                inps = self.input_preproc(inps, self.stats_inputs)
                outs = self.output_preproc(outs, self.stats_outputs)
                self.optimizer.zero_grad()
                loss = -self.model.log_prob(outs, context=inps).mean()
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.cpu().detach()*states.shape[0]
                total_inputs += states.shape[0]
            running_train_loss = running_train_loss/total_inputs
            if (epoch + 1) % int(epochs/5) == 0:
                progress = (epoch / epochs) * 100
                rounded_progress = round(progress / 20) * 20
                print(f'training {rounded_progress}% complete loss: {running_train_loss}')
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
                inps = self.input_preproc(inps, self.stats_inputs)
                outs = self.output_preproc(outs, self.stats_outputs)
                loss = 0
                for ei in range(self.ensemble_size):
                    kwargs = {'rand_mask': False, 'mask_index': ei}
                    comp_log_prob = self.model.log_prob(outs, context=inps, kwargs=kwargs)
                    loss += (torch.exp(comp_log_prob)*1/self.ensemble_size)
                loss = torch.log(loss)
                loss[loss.isinf()] = -250
                loss = -loss.mean().cpu().detach()
                running_loss += loss.cpu().detach()*states.shape[0]
                total_inputs += states.shape[0]
        running_loss = running_loss/total_inputs
        return running_loss 

    def detach_model(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def attach_model(self):
        for p in self.model.parameters():
            p.requires_grad = True


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        if self.fixed_masks:
            mask_path = path[:-3]+'_masks'+path[-3:]
            if not self.output_ensemble:
                torch.save(self.model._distribution._context_encoder.masks, mask_path)
            else:
                masks = []
                for i in range(len(self.model._transform._transforms)):
                    try:
                        masks.append(self.model._transform._transforms[i].autoregressive_net.masks)
                    except AttributeError:
                        try:
                            masks.append(self.model._transform._transforms[1].transform_net.masks)
                        except AttributeError:
                            continue
                torch.save(masks, mask_path)
        self.save_constants(path)

    def load_model(self, path): 
        import pdb; pdb.set_trace()

    def attach_last_layer(self):
        self.model._distribution._context_encoder.fc3.weight.requires_grad = True
        self.model._distribution._context_encoder.fc3.bias.requires_grad = True


    def sample(self, numb_samps, context, kwargs=None,
            ensemble = True, ensemble_size = 10):
        (output_hat, base_hat, base_mean, base_std) = (
            self.model.sample(numb_samps, context = context,
            kwargs=kwargs))
        return (output_hat, base_hat, base_mean, base_std)

    def sample_and_log_prob(self, numb_samps, context, kwargs=None,
            ensemble = True, ensemble_size = 10):
        (output_hat, nflows_log_prob, component_log_prob,
            base_log_prob, base_hat, base_mean, base_std) = (
            self.model.sample_and_log_prob(numb_samps, context = context,
            kwargs=kwargs, ensemble = ensemble, ensemble_size = ensemble_size))
        return (output_hat, base_log_prob, base_mean, base_std)

    def sample_gauss(self, num_samps, context=None):
        base_network = self.model._distribution._context_encoder
        out = base_network(context)
        mu = out[:,:int(out.shape[1]/2)]
        sig = out[:,int(out.shape[1]/2):]
        sig = torch.exp(sig)
        norm_rv = torch.distributions.normal.Normal(mu, sig)
        samp = norm_rv.sample([num_samps])
        samp = samp.permute(1, 0, 2)
        return samp, mu, sig

    def grad_last_layer(self, x, num_samps=1, bait=False):
        self.attach_last_layer()
        target_hyp, pred_mu, pred_sig = self.sample_gauss(num_samps, context=x)
        all_grads = []
        for j in range(num_samps):
            all_samp_grads = []
            for i in range(x.shape[0]):
                self.optimizer.zero_grad()
                mu_i = pred_mu[i,:].repeat(1, 1)
                sig_i = pred_sig[i,:].repeat(1, 1)
                criterion = torch.nn.GaussianNLLLoss(reduction='mean', full=False)
                loss = -criterion(target_hyp[i, j:j+1,:], mu_i, sig_i**2)
                loss.backward(retain_graph=True)
                last_layer_gradients = torch.cat([self.model._distribution._context_encoder.fc3.weight.grad.detach().clone().reshape(-1),self.model._distribution._context_encoder.fc3.bias.grad.detach().clone().reshape(-1)])
                all_samp_grads.append(last_layer_gradients)
            all_grads.append(torch.stack(all_samp_grads))
        self.optimizer.zero_grad()
        self.detach_model()
        all_grads = torch.stack(all_grads).permute(1,0,2)
        all_grads = all_grads.squeeze()
        return all_grads
