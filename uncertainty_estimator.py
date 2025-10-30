import numpy as np
import torch


class MixtureEntropyEstimator(object):
    ''' Only works for mixture of Gaussians
    but can be adapted. Assumes diagonal covariance matrices
    '''
    def __init__(self, estimator):
        self.estimator = estimator

    def upper_bound(self, samp, comps, mus, sigs, weights, conditional_ent):
        log_pdf_sum = 0
        for i in range(len(mus)):
            log_pdf_sum += (comps==i).sum()*np.log(weights[i])+\
                    torch.distributions.normal.Normal(mus[i,:], 
                    sigs[i,:]).log_prob(samp[(comps==i),:]).sum()
        ent_upper_bound = -log_pdf_sum/samp.shape[0]
        return ent_upper_bound

    def lower_bound(self, samp, comps, mus, sigs, weights, conditional_ent):
        log_pdf_sum = 0
        #the formula for entropy for multivariate gaussians
        Sigs = sigs**2
        det_term = (2*np.pi*np.exp(1)*Sigs).prod(2)
        log_term = 1/2*torch.log(det_term)
        ent_lower_bound = log_term.mean(1)
        return ent_lower_bound
    
    #def monte_carlo(self, samp, comps, mus, sigs, weights, conditional_ent):
    #    log_prob = np.log(mixture_pdf(samp, mus, sigs, weights))
    #    return -log_prob.mean() 
    
    def kl_div(self, mus, sigs):
        mus = mus.type(torch.float64)
        sigs = sigs.type(torch.float64)
        Sigs = sigs**2 
        # now assumes 3 dimensional input 
        Sigs = Sigs.permute(1,0,2)
        mus = mus.permute(1,0,2)
        tr_term = (Sigs[:,None,:,:]*(Sigs**-1)).sum(3)
        det_term = torch.log((Sigs/Sigs[:,None,:,:]).prod(3))
        quad_term = torch.einsum('ijkl->ijk',(mus - mus[:,None,:,:])**2/Sigs)
        return .5 * (tr_term + det_term + quad_term - mus.shape[2]) 
    
    def bhatt_div(self, mus, sigs):
        mus = mus.type(torch.float64)
        sigs = sigs.type(torch.float64)
        Sigs = sigs**2 
        # now assumes 3 dimensional input 
        Sigs = Sigs.permute(1,0,2)
        mus = mus.permute(1,0,2)
        mean_sig = (Sigs[:,None,:,:]+Sigs)/2
        quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2/mean_sig)
        log_term = torch.log(((mean_sig)/
                torch.sqrt(Sigs[:,None]*Sigs)).prod(3))
        return ((1/8)*quad_term+(1/2)*log_term)
    
    def bhatt_div_faster(self, mus, sigs):
        mus = mus.type(torch.float64)
        sigs = sigs.type(torch.float64)
        Sigs = sigs**2 
        # now assumes 3 dimensional input 
        Sigs = Sigs.permute(1,0,2)
        mus = mus.permute(1,0,2)
        mean_sig = (Sigs[:,None,:,:]+Sigs)/2
        quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2/mean_sig)
        log_term = torch.log(((mean_sig)/
                torch.sqrt(Sigs[:,None]*Sigs)).prod(3))
        return ((1/8)*quad_term+(1/2)*log_term)

    def wasserstein_dist(self, mus, sigs):
        Sigs = sigs**2 
        # now assumes 3 dimensional input 
        Sigs = Sigs.permute(1,0,2)
        sigs = sigs.permute(1,0,2)
        mus = mus.permute(1,0,2)
        quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2)
        tr_term = (Sigs[:,None,:,:]+Sigs-
                2*torch.sqrt(sigs[:,None,:,:]*Sigs*sigs[:,None,:,:])).sum(3)
        return quad_term+tr_term

    def pairwise_exp(self, div, weights):
        #return (torch.log((torch.exp(-div)*torch.tensor(weights)).sum(1))*
        #        torch.tensor(weights)).sum()
        # Assumes uniform weights 
        return (torch.log((torch.exp(-div)*weights[0]).sum(1))*
                weights[0]).sum(0)

    def kl_exp(self, samp, comps, mus, sigs, weights, conditional_ent):
        kl = self.kl_div(mus, sigs)
        pairwise_dist = self.pairwise_exp(kl, weights)
        return conditional_ent-pairwise_dist
    
    def bhatt_exp(self, samp, comps, mus, sigs, weights, conditional_ent):
        bhatt = self.bhatt_div(mus, sigs)
        pairwise_dist = self.pairwise_exp(bhatt, weights)
        return conditional_ent-pairwise_dist
    
    def bhatt_exp_faster(self, samp, comps, mus, sigs, weights, conditional_ent):
        bhatt = self.bhatt_div_faster(mus, sigs)
        pairwise_dist = self.pairwise_exp(bhatt, weights)
        return conditional_ent-pairwise_dist
    
    def wasserstein_exp(self, samp, comps, mus, sigs, weights, conditional_ent):
        wass_2 = self.wasserstein_dist(mus, sigs)
        pairwise_dist = self.pairwise_exp(wass_2, weights)
        return conditional_ent-pairwise_dist
    
    def kde(self, samp, comps, mus, sigs, weights, conditional_ent):
        sigs = sigs.permute(1,0,2)
        mus = mus.permute(1,0,2)
        dist = torch.distributions.normal.Normal(mus, sigs)
        log_probs = dist.log_prob(mus[:,None,: ,:]).sum(3)
        kde_estimate = -self.pairwise_exp(-log_probs, weights)
        return kde_estimate
    
    def elk(self, samp, comps, mus, sigs, weights, conditional_ent):
        sigs = sigs.permute(1,0,2)
        mus = mus.permute(1,0,2)
        sigs = torch.sqrt(sigs[:,None,:,:]**2+sigs**2)
        numb_comp = mus.shape[0]
        mus_repeated = mus.repeat_interleave(numb_comp,0).reshape(numb_comp, 
                numb_comp, mus.shape[1], -1) 
        dist = torch.distributions.normal.Normal(mus_repeated, sigs)
        log_probs = dist.log_prob(mus).sum(3)
        elk_estimate = -self.pairwise_exp(-log_probs, weights)
        return elk_estimate
    
    def taylor_series(self, a):
        print('need to code')
    
    def estimate_entropy(self, samp, comps, mus, sigs, weights, conditional_ent):
        '''mus = (NxMxD)
        where N is the number of model inputs,
        M is the number of mixture components,
        D is the number of dimensions.
        '''
        entropy_func = eval('self.'+self.estimator)
        return entropy_func(samp, comps, mus, sigs, weights, conditional_ent)

class EpistemicUncertaintyEstimator(MixtureEntropyEstimator):
    
    def kl_mean(self, samp, comps, mus, sigs, weights, conditional_ent):
        kl = self.kl_div(mus, sigs)
        kl =  kl.reshape(-1, kl.shape[2])
        kl[kl==0] = torch.nan
        return kl.nanmean(0) 
    
    def kl_max(self, samp, comps, mus, sigs, weights, conditional_ent):
        kl = self.kl_div(mus, sigs)
        kl =  kl.reshape(-1, kl.shape[2])
        kl = kl.nan_to_num(-50)
        return kl.max(0)[0]
    
    def wasserstein_mean(self, samp, comps, mus, sigs, weights, conditional_ent):
        wass_2 = self.wasserstein_dist(mus, sigs)
        wass_2 =  wass_2.reshape(-1, wass_2.shape[2])
        wass_2[wass_2==0] = torch.nan
        return wass_2.nanmean(0) 
    
    def wasserstein_max(self, samp, comps, mus, sigs, weights, conditional_ent):
        wass_2 = self.wasserstein_dist(mus, sigs)
        wass_2 =  wass_2.reshape(-1, wass_2.shape[2])
        wass_2 = wass_2.nan_to_num(-50)
        return wass_2.max(0)[0]

    def estimate_epi_uncertainty(self, samp, comps, mus, sigs, weights, method='default'):
        if method == 'default':
            mi = self.estimate_entropy(samp, comps, mus, sigs, weights, 0)
        elif method != 'default':
            entropy_func = eval('self.'+method)
            mi = entropy_func(samp, comps, mus, sigs, weights, 0)
        return mi 
