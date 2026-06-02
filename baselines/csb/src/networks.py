import pandas as pd
import numpy as np
import os
#from news.process_news import get_v

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import matplotlib.pyplot as plt


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, z_dim, t_dim_latent, attn_dim):
        super().__init__()
        self.query = nn.Linear(z_dim, attn_dim)
        self.key = nn.Linear(t_dim_latent, attn_dim)
        self.value = nn.Linear(t_dim_latent, attn_dim)
        self.scale = attn_dim ** 0.5

    def forward(self, z, t_latent):
        # z: [batch, z_dim], t_latent: [batch, t_dim_latent]
        Q = self.query(z)            # [batch, attn_dim]
        K = self.key(t_latent)       # [batch, attn_dim]
        V = self.value(t_latent)     # [batch, attn_dim]
        
        attn_weights = torch.softmax((Q * K).sum(dim=1, keepdim=True) / self.scale, dim=1)
        attended = attn_weights * V  # element-wise attention application
        
        return attended  # [batch, attn_dim]


class TorchBSplineBasis(nn.Module):
    def __init__(self, num_basis=10, degree=3, treatment_range=(0.0, 1.0), input_dim=1):
        super().__init__()
        self.num_basis = num_basis
        self.degree = degree
        self.treatment_range = treatment_range
        self.input_dim = input_dim

        # Create knots for each input dimension
        self.knots_list = nn.ParameterList()
        for _ in range(input_dim):
            n_knots = num_basis + degree + 1
            knots = torch.linspace(treatment_range[0], treatment_range[1], n_knots - 2 * degree)
            start = knots[0].repeat(degree)
            end = knots[-1].repeat(degree)
            full_knots = torch.cat([start, knots, end])
            self.register_buffer(f"knots_{_}", full_knots)
            self.knots_list.append(getattr(self, f"knots_{_}"))

    def basis_function(self, t, i, k, knots):
        if k == 0:
            return ((knots[i] <= t) & (t < knots[i + 1])).float()
        else:
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            term1 = 0.0 if denom1 == 0 else (t - knots[i]) / denom1 * self.basis_function(t, i, k - 1, knots)
            term2 = 0.0 if denom2 == 0 else (knots[i + k + 1] - t) / denom2 * self.basis_function(t, i + 1, k - 1, knots)
            return term1 + term2

    def compute_basis(self, t, knots):
        batch_size = t.shape[0]
        basis = []
        for i in range(self.num_basis):
            b = self.basis_function(t, i, self.degree, knots)
            basis.append(b.unsqueeze(1))
        return torch.cat(basis, dim=1)  # (batch_size, num_basis)

    def forward(self, t):
        """
        Args:
            t: shape (batch_size, input_dim)
        Returns:
            basis: shape (batch_size, num_basis ** input_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)

        #assert t.shape[1] == self.input_dim, "Input treatment dimension mismatch"

        bases = []
        for d in range(self.input_dim):
            basis_d = self.compute_basis(t[:, d], self.knots_list[d])  # (batch, num_basis)
            bases.append(basis_d)

        if self.input_dim == 1:
            return bases[0]
        else:
            # Tensor product of basis functions
            # Start with shape: (batch, num_basis)
            basis = bases[0]
            for b in bases[1:]:
                basis = torch.einsum('bi,bj->bij', basis, b).reshape(basis.shape[0], -1)
            return basis  # shape: (batch, num_basis ** input_dim)

class CCS_Counterfactual_Net(nn.Module):
    def __init__(self, x_dim, t_dim_latent, z_dim, y_dim, t_input_dim = 1, hidden_dim=512, hidden_dim_t=16, attn_dim=64,
                 use_attention=True, use_spline=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_spline = use_spline

        self.encoder_z = nn.Sequential(
            MLPBlock(x_dim, hidden_dim),
            MLPBlock(hidden_dim, hidden_dim),
            MLPBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, z_dim)
        )

        if self.use_spline:
            self.spline_encoder = TorchBSplineBasis(num_basis=10, degree=3, treatment_range=(0.0, 1.0))
            t_input_dim = 10  # B-spline basis size
        else:
            t_input_dim = t_input_dim  # Raw treatment input dimension (assumed scalar)

        self.encoder_t = nn.Sequential(
            nn.Linear(t_input_dim, hidden_dim_t),
            nn.ReLU(),
            nn.Linear(hidden_dim_t, t_dim_latent)
        )

        self.decoder_t = nn.Sequential(
            MLPBlock(z_dim, hidden_dim_t),
            nn.Linear(hidden_dim_t, 1)  # Assuming t is scalar
        )

        if self.use_attention:
            self.attention = Attention(z_dim, t_dim_latent, attn_dim)
            decoder_y_input_dim = z_dim + attn_dim
        else:
            decoder_y_input_dim = z_dim

        self.decoder_y = nn.Sequential(
            MLPBlock(decoder_y_input_dim, hidden_dim),
            MLPBlock(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, y_dim)
        )

    def forward(self, x, t):
        z = self.encoder_z(x)  # z | x

        if self.use_spline:
            t_feats = self.spline_encoder(t)  # Spline features
        else:
            t_feats = t  # Raw treatment input

        t_latent = self.encoder_t(t_feats)     # Latent representation of t
        t_logits = self.decoder_t(z)           # Reconstruct t from z

        if self.use_attention:
            attn_out = self.attention(z, t_latent)
            z_attn = torch.cat([z, attn_out], dim=1)
        else:
            z_attn = z

        y_pred = self.decoder_y(z_attn)
        return z, t_latent, y_pred
