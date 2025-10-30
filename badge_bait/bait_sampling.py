import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
import gc
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch.cuda.amp import autocast
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances


def select(X, K, fisher, iterates, lamb=1, nLabeled=0):

    numEmbs = len(X)
    indsAll = []
    dim = X.shape[-1]
    rank = X.shape[-2]

    currentInv = torch.inverse(lamb * torch.eye(dim).cuda() + iterates.cuda() * nLabeled / (nLabeled + K)).half()
    X = X * np.sqrt(K / (nLabeled + K))
    fisher = fisher.cuda().half()

    # forward selection, over-sample by 2x
    #print('forward selection...', flush=True)
    over_sample = 2
    with autocast():
        for i in range(int(over_sample *  K)):

            # check trace with low-rank updates (woodbury identity)
            xt_ = X.cuda() 
            innerInv = torch.inverse((torch.eye(rank).cuda().half() + xt_ @ currentInv @ xt_.transpose(1, 2)).float()).half().detach()
            innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
            traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

            # clear out gpu memory
            xt = xt_.cpu()
            del xt, innerInv
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            # get the smallest unselected item
            traceEst = traceEst.detach().cpu().numpy()
            for j in np.argsort(traceEst)[::-1]:
                if j not in indsAll:
                    ind = j
                    break

            indsAll.append(ind)
            #print(i, ind, traceEst[ind], flush=True)
           
            # commit to a low-rank update
            xt_ = X[ind].unsqueeze(0).cuda()
            innerInv = torch.inverse(torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        # backward pruning
        #print('backward pruning...', flush=True)
        for i in range(len(indsAll) - K):

            # select index for removal
            xt_ = X[indsAll].cuda()
            innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
            traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
            delInd = torch.argmin(-1 * traceEst).item()
            #print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)


            # low-rank update (woodbury identity)
            xt_ = X[indsAll[delInd]].unsqueeze(0).cuda()
            innerInv = torch.inverse(-1 * torch.eye(rank).cuda() + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

            del indsAll[delInd]

        del xt_, innerInv, currentInv
        torch.cuda.empty_cache()
        gc.collect()
    return indsAll

def BaitSampling(grad_exp_embedding, grad_exp_embedding_train, n, train_size):
    xt = grad_exp_embedding.half() 

    # get fisher
    #print('getting fisher matrix...', flush=True)
    batchSize = 100 # should be as large as gpu memory allows
    fisher = torch.zeros(xt.shape[-1], xt.shape[-1])    
    for i in range(int(np.ceil(grad_exp_embedding.shape[0] / batchSize))):
        xt_ = xt[i * batchSize : (i + 1) * batchSize].cuda()
        op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
        fisher = fisher + op
        xt_ = xt_.cpu()
        del xt_, op
        torch.cuda.empty_cache()
        gc.collect()
    # get fisher only for samples that have been seen before
    init = torch.zeros(xt.shape[-1], xt.shape[-1])
    xt2 = grad_exp_embedding_train.half()
    for i in range(int(np.ceil(len(xt2) / batchSize))):
        xt_ = xt2[i * batchSize : (i + 1) * batchSize].cuda()
        op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt2)), 0).detach().cpu()
        init = init + op
        xt_ = xt_.cpu()
        del xt_, op
        torch.cuda.empty_cache()
        gc.collect()

    chosen = select(xt, n, fisher, init, lamb=0.01, nLabeled=train_size)
    return chosen
