import re
import random
import os
import sys
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import numpy.linalg as la
from numpy import log
import scipy
from scipy.stats import norm, rv_continuous
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree
from nflows_ensemble_model import nflows_ensemble
from pens_model import pens
from mc_drop_model import mc_drop


### CONTINUOUS ESTIMATORS


def entropy_utils(x, k=3, base=np.e):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    return (const + n_features * np.log(nn).mean()) / log(base)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')
###

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class LogitNormal(rv_continuous):
    def __init__(self, scale=1, loc=0):
        super().__init__(self)
        self.scale = scale
        self.loc = loc

    def _pdf(self, x):
        return norm.pdf(logit(x), loc=self.loc, scale=self.scale)/(x*(1-x))

def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.

    - accepts stacks of means, but only one S0 and S1

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    S0 = np.diag(S0)
    S1 = np.diag(S1)
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    print(f'Trace Term: {tr_term}')
    print(f'Determinant Term: {det_term}')
    print(f'Quadratic Term: {quad_term}')
    return .5 * (tr_term + det_term + quad_term - N)

def Wasserstein_GP(mu_0, K_0, mu_1, K_1):

    K_0 = np.diag(K_0)
    K_1 = np.diag(K_1)
    sqrtK_0 = scipy.linalg.sqrtm(K_0)
    first_term = np.dot(sqrtK_0, K_1)
    K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

    cov_dist = np.trace(K_0) + np.trace(K_1) - 2 * np.trace(scipy.linalg.sqrtm(K_0_K_1_K_0))
    l2norm = (np.sum(np.square(abs(mu_0 - mu_1))))
    d = np.real(np.sqrt(l2norm + cov_dist))

    return d

def bhatt_dist(mu0, sig0, mu1, sig1):
    diff = mu0-mu1
    mean_sig = (sig0+sig1)/2
    quad_term = (diff*mean_sig**-1*diff).sum()
    log_term = torch.log((mean_sig.prod())/(torch.sqrt(sig0.prod()*sig1.prod())))
    print(f'quad term: {quad_term/8}')
    print(f'log term: {log_term/2}')
    return (1/8)*quad_term+(1/2)*log_term

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 


def split_mean_kls(all_kls): 
    num_kls = len(all_kls[0]) 
    reformated = [] 
    means = [] 
    for i in range(num_kls): 
        reformed = [np.array(j[i]) for j in all_kls] 
        reformed = np.array(reformed) 
        reformated.append(reformed) 
        mean = reformed[np.isfinite(reformed).any(axis=1)].mean(0) 
        means.append(mean) 
    return reformated, means 

def gen_folder_uncertain(args):
    branch_folder = args.env
    sub_branch_folder = args.model
    if sub_branch_folder in ['nflows_ensemble', 'nn_ensemble']:
        sub_branch_folder += '_fixedmasks'
    if args.test_num_samples:
        branch_folder += '_test_numsamples'
        sub_branch_folder += f'_{args.acquisition_type}_numb_samps_{args.numb_samps}'
    sub_branch_folder += '_'+args.acquisition_function
    sub_branch_folder += '_seed'+str(args.seed)
    return branch_folder, sub_branch_folder

def normalize(states, stats):
    return (states-stats[0])/(stats[1]-stats[0]+1e-6)

def standardize(states, stats):
    return (states-stats[2])/(stats[3])+1e-6

def un_normalize(preds, stats):
    return preds*(stats[1]-stats[0]+1e-6)+stats[0]
    
def un_standardize(preds, stats):
    return preds*(stats[3]+1e-6)+stats[2]

def identity(preds, stats):
    return preds

def instantiate_model(args, output_dim, context_dim, device,
        input_preproc, output_preproc):
    if args.model == 'nflows_ensemble':
        model = nflows_ensemble(args.num_layers, args.hids, output_dim, context_dim,
                    10, 1.2, args.lr, device, input_preproc, output_preproc,
                    rqs = args.rqs, ensemble_size=args.ensemble_size, base_ensemble=args.base_distro)
    elif args.model == 'mc_drop':
        model = mc_drop(args.num_layers, args.hids, output_dim, context_dim,
            args.lr, device, input_preproc, output_preproc)
    elif args.model == 'nn_ensemble':
        model = pens(args.num_layers, args.hids, output_dim, context_dim,
            args.lr, device, input_preproc, output_preproc,
            multihead = False, fixed_masks = True,
            ensemble_size = args.ensemble_size)
    return model

