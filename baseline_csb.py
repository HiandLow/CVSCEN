import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add ibex to sys path so we can import its modules
current_dir = os.path.dirname(os.path.abspath(__file__))
csb_dir = os.path.join(current_dir, "baselines", "csb")
if csb_dir not in sys.path:
    sys.path.insert(0, csb_dir)

from CCS_divergence import CS
from src.networks import CCS_Counterfactual_Net

class NumpyDataset(Dataset):
    def __init__(self, X, T, Y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        if len(self.T.shape) == 1:
            self.T = self.T.unsqueeze(1)
        if Y is not None:
            self.Y = torch.tensor(Y, dtype=torch.float32)
            if len(self.Y.shape) == 1:
                self.Y = self.Y.unsqueeze(1)
        else:
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.T[idx], self.X[idx], self.Y[idx]
        else:
            return self.T[idx], self.X[idx]

class CSBWrapper:
    def __init__(self, num_features=25):
        self.num_features = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters (from IBEX main.py defaults)
        self.batch_size = 64
        self.n_epochs = 200 # Usually ~3000 in original paper, but keeping 200 for benchmark fairness
        self.lr = 1e-3
        self.beta = 0.001
        self.gamma = 0.1
        self.use_attention = True
        self.use_spline = False
        
        # Init model
        t_dim_latent = 8
        z_dim = 32  # Used 32 for News/IHDP style, 16 for Mimic
        hidden_dim = 512
        
        self.model = CCS_Counterfactual_Net(
            x_dim=self.num_features,
            t_dim_latent=t_dim_latent,
            z_dim=z_dim,
            y_dim=1,
            hidden_dim=hidden_dim,
            hidden_dim_t=t_dim_latent,
            attn_dim=64,
            use_attention=self.use_attention,
            use_spline=self.use_spline,
        ).to(self.device)

    def fit(self, X, T, Y):
        dataset = NumpyDataset(X, T, Y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.n_epochs):
            for t, x, y in loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                
                z, t_logits, y_pred = self.model(x, t)

                # 1) Outcome reconstruction loss
                loss_y = criterion(y_pred, y)

                # 2) Independence regularisation via Contrastive Score (CS)
                joint_samples = torch.cat((z, t), dim=1)
                z_perm = z[torch.randperm(z.size(0))]
                indep_samples = torch.cat((z_perm, t), dim=1)
                loss_cs = CS(indep_samples, joint_samples)

                # 3) Latent space regularisation
                loss_reg = torch.norm(z, dim=0).sum()

                loss = loss_y + self.beta * loss_cs + self.gamma * loss_reg

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

    def predict(self, X, T):
        self.model.eval()
        
        if np.isscalar(T):
            T = np.full(X.shape[0], T)
            
        dataset = NumpyDataset(X, T)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for t_batch, x_batch in loader:
                x_batch = x_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                
                _, _, y_pred = self.model(x_batch, t_batch)
                preds.append(y_pred.cpu().numpy())
                
        return np.concatenate(preds).flatten()
