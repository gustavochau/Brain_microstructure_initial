import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance
import torch
import numpy as np
import normflows as nf
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from scipy.spatial import distance
import yaml
import scipy
from scipy import stats

import matplotlib.pyplot as plt
import os
import random

from scipy.spatial import distance
import scipy.stats as scis
import scipy.io as scio
import pickle 
from numpy import mean
from numpy import var
from math import sqrt
import pandas as pd

def emd (a,b):
    earth = 0
    earth1 = 0
    diff = 0
    s= len(a)
    su = []
    diff_array = []
    for i in range (0,s):
        diff = a[i]-b[i]
        diff_array.append(diff)
        diff = 0
    for j in range (0,s):
        earth = (earth + diff_array[j])
        earth1= abs(earth)
        su.append(earth1)
    emd_output = sum(su)/(s-1)
    return emd_output

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return u1, u2, u1-u2, (u1 - u2) / s

# Generate joint prob densities for all samples (multiple voxels per region)
def gen_2dprobs(model,x_all,num_mri_raw, device):
    model.eval()
    delta_g = 0.01
    delta_d = 0.02
    bins_g = np.arange(0,1.01,delta_g)
    bins_d = np.arange(0,2.02,delta_d)
    curve_g = np.arange(0,1,delta_g).astype('float32')
    curve_d = np.arange(0,2,delta_d).astype('float32')
    probs2d = []
    for selected_index in range(x_all.shape[0]):
        xx, yy = torch.meshgrid(torch.from_numpy(curve_d), torch.from_numpy(curve_g), indexing='ij')
        zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
        zz = zz.to(device)
        
        selected_input = torch.from_numpy(np.expand_dims(x_all[selected_index,:].astype('float32'),0)).to(device)
        selected_context = model.context_encoder(selected_input, num_mri_raw).to(device)
        rep_context = (selected_context + torch.zeros((zz.shape[0],1)).to(device))

        log_prob = model.flow_model.log_prob(zz, rep_context).to('cpu').view(*xx.shape)
        prob = torch.exp(log_prob)
        prob[torch.isnan(prob)] = 0
        probs2d.append(prob.data.numpy())
    return np.array(probs2d)

# Generate mean probability distributions taken from all the voxels in a region
def average_2dprobs(probs2d, mouse_nums, mouse_regions, mouse_nums_add, mouse_regions_add, mri_gratios):
    meansJoint = []
    meansMRIGratios = []
    for num, region in zip(mouse_nums, mouse_regions):
        sel_ind = np.bitwise_and(mouse_nums_add==num,mouse_regions_add==region)
        #print(f'sel_ind:{np.sum(sel_ind)}')
        mean_prob2d = np.mean(probs2d[sel_ind,:,:], axis=0)
        mean_g = np.mean(mri_gratios[sel_ind])
        meansJoint.append(mean_prob2d)
        meansMRIGratios.append(mean_g)
    return np.array(meansJoint), np.array(meansMRIGratios)

def evaluate_jsd(means_prob2d,y_all): 
    delta_g = 0.01
    delta_d = 0.02
    bins_g = np.arange(0,1+delta_g,delta_g)
    bins_d = np.arange(0,2.02,delta_d)
    curve_g = np.arange(0,1,delta_g).astype('float32')
    curve_d = np.arange(0,2,delta_d).astype('float32')
    distances_diameter = []
    wasserstein_diameter = []
    kldivergence_diameter = []
    distances_gratio = []
    wasserstein_gratio = []
    kldivergence_gratio = []
    
    for selected_index in range(means_prob2d.shape[0]):
        
        prob = means_prob2d[selected_index]

        # integrates to get the marginal densities
        p_g = np.sum(prob,axis=0)*delta_d
        p_d = np.sum(prob,axis=1)*delta_g

        hist_d,_ = np.histogram(y_all[selected_index,:,0], bins=bins_d, density=True)
        distances_diameter.append(distance.jensenshannon(p_d, hist_d))
        wasserstein_diameter.append(emd (list(p_d.flatten()),list(hist_d.flatten())))
        kldivergence_diameter.append(scipy.stats.entropy(pk=(p_d.flatten()+1E-8), qk=(hist_d.flatten()+1E-8)))
        
        hist_g,_ = np.histogram(y_all[selected_index,:,1], bins=bins_g, density=True)
        distances_gratio.append(distance.jensenshannon(p_g, hist_g))
        wasserstein_gratio.append(emd (list(p_g.flatten()),list(hist_g.flatten())))
        kldivergence_gratio.append(scipy.stats.entropy(pk=(p_g.flatten()+1E-8), qk=(hist_g.flatten()+1E-8)))
        
    return np.mean(distances_diameter), np.mean(wasserstein_diameter), np.mean(kldivergence_diameter), np.mean(distances_gratio), np.mean(wasserstein_gratio), np.mean(kldivergence_gratio)

def eval_dist_reference(y_all):
    delta_g = 0.01
    delta_d = 0.02
    bins_g = np.arange(0,1+delta_g,delta_g)
    bins_d = np.arange(0,2.02,delta_d)
    curve_g = np.arange(0,1,delta_g).astype('float32')
    curve_d = np.arange(0,2,delta_d).astype('float32')
    
    distances_diameter = []
    wasserstein_diameter = []
    kldivergence_diameter = []

    distances_gratio = []
    wasserstein_gratio = []
    kldivergence_gratio = []
    
    for selected_index in range(y_all.shape[0]):
        curr_d_samples = y_all[selected_index,:,0]
        curr_g_samples = y_all[selected_index,:,1]
        
        fit_alpha, fit_loc, fit_beta=stats.gamma.fit(curr_d_samples)
        p_d = stats.gamma.pdf(curve_d, a=fit_alpha, loc=fit_loc, scale=fit_beta)
        
        fit_alpha, fit_loc, fit_beta=stats.gamma.fit(curr_g_samples)
        p_g = stats.gamma.pdf(curve_g, a=fit_alpha, loc=fit_loc, scale=fit_beta)
        
        hist_d,_ = np.histogram(curr_d_samples, bins=bins_d, density=True)
        distances_diameter.append(distance.jensenshannon(p_d, hist_d))
        wasserstein_diameter.append(emd (list(p_d.flatten()),list(hist_d.flatten())))
        kldivergence_diameter.append(scipy.stats.entropy(pk=(p_d.flatten()+1E-8), qk=(hist_d.flatten()+1E-8)))
        
        hist_g,_ = np.histogram(curr_g_samples, bins=bins_g, density=True)
        distances_gratio.append(distance.jensenshannon(p_g, hist_g))
        wasserstein_gratio.append(emd (list(p_g.flatten()),list(hist_g.flatten())))
        kldivergence_gratio.append(scipy.stats.entropy(pk=(p_g.flatten()+1E-8), qk=(hist_g.flatten()+1E-8)))
    
    return np.mean(distances_diameter), np.mean(wasserstein_diameter), np.mean(kldivergence_diameter), np.mean(distances_gratio), np.mean(wasserstein_gratio), np.mean(kldivergence_gratio)
       

def evaluate_error(means_prob2d, mouse_num_all, regions_all, base_df):
    delta_g = 0.01
    delta_d = 0.02
    bins_g = np.arange(0,1+delta_g,delta_g)
    bins_d = np.arange(0,2.02,delta_d)
    curve_g = np.arange(0,1,delta_g).astype('float32')
    curve_d = np.arange(0,2,delta_d).astype('float32')
    all_dataframes = base_df.copy()
    all_dataframes['genu']['ML_mean'] = np.nan
    all_dataframes['body']['ML_mean'] = np.nan
    all_dataframes['splenium']['ML_mean'] = np.nan
    
    for selected_index in range(means_prob2d.shape[0]):
        
        prob = means_prob2d[selected_index]
        # integrates to get the marginal densities
        p_g = np.sum(prob,axis=0)*delta_d
        mean_dl = np.sum(curve_g*p_g)/np.sum(p_g)
        
        # data to df
        region = regions_all[selected_index]
        mouse_num = mouse_num_all[selected_index]
        df_index = base_df[region].index[base_df[region]['mouse_num'] == mouse_num].tolist()[0]
        all_dataframes[region].at[df_index,'ML_mean'] = mean_dl
    return all_dataframes

def get_group_metrics(all_dataframes, regions, with_test = False):
    all_region_group_analysis = []
    for i, region in enumerate(regions):
        df = all_dataframes[region].dropna(subset=["em_d_mean"])
        # obtains wt and het tables
        if with_test:
            wt_df = df[df['group']=='wt']
            het_df = df[df['group']=='het']
        else:
            wt_df = df[((df['set']=='Train') | (df['set']=='Val')) & (df['group']=='wt')]
            het_df = df[((df['set']=='Train') | (df['set']=='Val')) & (df['group']=='het')]
        # EM comparison
        em_wt = np.array(wt_df['em_g_mean'])
        em_het = np.array(het_df['em_g_mean'])
        u_wt_em, u_het_em, diff_means_em, cohen_em = cohend(em_wt, em_het)
        # EM comparison
        mri_wt = np.array(wt_df['ML_mean'])
        mri_het = np.array(het_df['ML_mean'])
        u_wt_mri, u_het_mri, diff_means_mri, cohen_mri = cohend(mri_wt, mri_het)
        row = [region, u_wt_em, u_het_em, diff_means_em, cohen_em, u_wt_mri, u_het_mri, diff_means_mri, cohen_mri]
        all_region_group_analysis.append(row)
    all_region_group_analysis = pd.DataFrame(all_region_group_analysis)
    all_region_group_analysis.columns = ['Region','EM_mean_wt','EM_mean_het','EM_diff_means','EM_cohen',
                                         'ML_mean_wt','ML_mean_het','ML_diff_means','ML_cohen']
    return all_region_group_analysis 

