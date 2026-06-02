# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 18:20:15 2023

@author: 61995
"""


import numpy as np
import torch


def GaussianMatrix(X,Y,sigma):
    size1 = X.size()
    size2 = Y.size()
    G = (X*X).sum(-1)
    H = (Y*Y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1,size2[0])
    R = H.unsqueeze(-1).T.repeat(size2[0],1)
    
    
    H = Q + R - 2*X@(Y.T)
    H = torch.exp(-H/2/sigma**2)
    
    
    return H


def CSD_4(x1, x2, y1, y2, sigma=2):
    # Do NOT wrap with torch.tensor() – this detaches them from autograd
    K1 = GaussianMatrix(x1, x1, sigma)
    K2 = GaussianMatrix(x2, x2, sigma)
    
    L1 = GaussianMatrix(y1, y1, sigma)
    L2 = GaussianMatrix(y2, y2, sigma)
    
    K12 = GaussianMatrix(x1, x2, sigma)
    L12 = GaussianMatrix(y1, y2, sigma)
    
    K21 = GaussianMatrix(x2, x1, sigma)
    L21 = GaussianMatrix(y2, y1, sigma)

    H1 = K1 * L1
    self_term1 = (H1.sum(-1) / (K1.sum(-1)**2)).sum(0)

    H2 = K2 * L2
    self_term2 = (H2.sum(-1) / (K2.sum(-1)**2)).sum(0)

    H3 = K12 * L12
    cross_term1 = (H3.sum(-1) / (K1.sum(-1) * K12.sum(-1))).sum(0)

    H4 = K21 * L21
    cross_term2 = (H4.sum(-1) / (K2.sum(-1) * K21.sum(-1))).sum(0)

    cs1 = -2 * torch.log2(cross_term1) + torch.log2(self_term1) + torch.log2(self_term2)
    cs2 = -2 * torch.log2(cross_term2) + torch.log2(self_term1) + torch.log2(self_term2)

    return (cs1 + cs2) / 2  # <-- This is a scalar tensor


def CS(x1,x2,sigma = 1): # marginal cs divergence
    # x1 = torch.tensor(x1)
    # x2 = torch.tensor(x2)
    
    K1 = GaussianMatrix(x1,x1,sigma)
    K2 = GaussianMatrix(x2,x2,sigma)

    
    K12 = GaussianMatrix(x1,x2,sigma)


    dim1 = K1.shape[0]
    self_term1 = K1.sum()/(dim1**2)
    
    dim2 = K2.shape[0]
    self_term2 = K2.sum()/(dim2**2)
    
    cross_term = K12.sum()/(dim1*dim2)
    
    cs =  -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)
   
    return cs#.item()