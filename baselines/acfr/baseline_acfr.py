import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy
from tqdm import trange

# Add the cloned repository to sys.path to resolve internal imports
sys.path.insert(0, os.path.dirname(__file__))

from models.acfr import acfr

class NumpyDataset(Dataset):
    def __init__(self, X, T, Y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.T = torch.tensor(T, dtype=torch.float32)
        if Y is not None:
            self.Y = torch.tensor(Y, dtype=torch.float32)
        else:
            self.Y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return self.T[idx], self.X[idx], self.Y[idx]
        else:
            return self.T[idx], self.X[idx]

class ACFRWrapper:
    def __init__(self, num_features=25, **kwargs):
        self.num_features = num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim = kwargs.get('dim_layer', 50)
        self.cfg = {
            'lr1': kwargs.get('lr', 0.005),
            'lr2': kwargs.get('lr_s', 0.05),
            'gamma1': kwargs.get('gamma1', 1),
            'gamma2': kwargs.get('gamma2', 0.2),
            'm': dim,
            'std': kwargs.get('std', 0.2),
            'batch_size': kwargs.get('batch_size', 64),
            'weight_decay': kwargs.get('weight_decay', 0.001),
            'epoch_number': kwargs.get('epoch_total', 300),
            'encoder_net': {'input_dim': self.num_features, 'hidden_dims': [dim, dim], 'output_dim': dim},
            'prediction_net': {'input_dim': dim, 'hidden_dims': [dim, dim], 'output_dim': 1},
            'discrimination_net': {'input_dim': dim, 'hidden_dims': [dim, dim], 'output_dim': 1}
        }

    def fit(self, X, T, Y):
        # We will split a small validation set internally for early stopping / best model selection
        val_size = int(0.1 * len(X))
        indices = torch.randperm(len(X))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_dataset = NumpyDataset(X[train_indices], T[train_indices], Y[train_indices])
        val_dataset = NumpyDataset(X[val_indices], T[val_indices], Y[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=self.cfg['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.cfg['batch_size'], shuffle=False)

        self.model = acfr(self.cfg).to(self.device)
        self.model.train()

        best_val = 1e10
        best_model_state = None

        n_epoch = self.cfg['epoch_number']

        for epoch in range(1, n_epoch + 1):
            self.model.train()
            for batch_idx, (t_batch, x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # ACFR specific training loop
                t_hat = self.model.forward_D(x_batch)
                l_t, err = self.model.backward_D(t_batch, t_hat)
                
                y_hat1, t_hat, y_hat2 = self.model.forward_G(x_batch, t_batch, err)
                l = self.model.backward_G(t_batch, y_batch, t_hat, y_hat1, y_hat2, err)

            # Validation
            val_loss = self._val(val_loader)
            if val_loss < best_val and epoch > n_epoch / 2:
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_val = val_loss

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def _val(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for t_batch, x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                y_hat = self.model(x_batch, t_batch)
                loss = F.mse_loss(y_batch, torch.squeeze(y_hat, 1))
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def predict(self, X, T):
        self.model.eval()
        
        # If T is a scalar, broadcast it
        import numpy as np
        if np.isscalar(T):
            T = np.full(X.shape[0], T)
            
        test_dataset = NumpyDataset(X, T)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg['batch_size'], shuffle=False)
        
        preds = []
        with torch.no_grad():
            for t_batch, x_batch in test_loader:
                x_batch = x_batch.to(self.device)
                t_batch = t_batch.to(self.device)
                y_hat = self.model(x_batch, t_batch)
                preds.append(y_hat.cpu().numpy())
                
        return np.concatenate(preds).flatten()
