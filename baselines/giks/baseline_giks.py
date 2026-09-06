import sys
import os
import numpy as np
import torch
import torch.nn as nn


class GIKSWrapper:
    """
    Wrapper for the GIKS (Gradient Interpolation and Kernel Smoothing) model
    from Nagalapatti et al., AAAI 2024.
    
    Uses a VCNet backbone with GIKS-style training:
    1. Factual training (standard MSE on observed data)
    2. Gradient Interpolation (GI) for nearby counterfactuals
    """
    def __init__(self, num_features, **kwargs):
        self.num_features = num_features
        self.hparams = kwargs
        
        # Setup paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        giks_dir = current_dir
        
        # Temporarily swap sys.modules to avoid 'utils' collision
        # Our project has utils.py, GIKS has utils/ package
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == 'utils' or key.startswith('utils.'):
                saved_modules[key] = sys.modules.pop(key)
        
        # Also temporarily remove our project dir from sys.path head
        # so that GIKS's utils/ package is found first
        project_dir = os.path.dirname(os.path.abspath(__file__))
        path_was_first = (sys.path[0] == project_dir) if sys.path else False
        if project_dir in sys.path:
            sys.path.remove(project_dir)
        
        if giks_dir not in sys.path:
            sys.path.insert(0, giks_dir)
        
        # Patch the device function BEFORE importing anything else
        import utils.common_utils as cu
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        cu.get_device = lambda: self._device
        
        # Import the model architecture
        import continuous.dynamic_net as DNet
        from continuous.data_helper import Dataset_from_matrix
        
        self.DNet = DNet
        self.Dataset_from_matrix = Dataset_from_matrix
        
        # Restore project dir to sys.path
        if project_dir not in sys.path:
            if path_was_first:
                sys.path.insert(0, project_dir)
            else:
                sys.path.append(project_dir)
        
        # Restore our utils module (but keep GIKS utils cached too under different keys)
        for key, mod in saved_modules.items():
            sys.modules[key] = mod
        
    def fit(self, X, T, Y):
        """
        Train VCNet backbone with GIKS augmentation.
        X: numpy array (n, d) - covariates
        T: numpy array (n,) - treatments in [0,1]
        Y: numpy array (n,) - outcomes
        """
        n_samples = X.shape[0]
        indim = self.num_features
        
        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float64).to(self._device)
        T_t = torch.tensor(T, dtype=torch.float64).to(self._device)
        Y_t = torch.tensor(Y, dtype=torch.float64).to(self._device)
        
        # Build train_matrix: [treatment, features, outcome]
        train_matrix = torch.cat([
            T_t.view(-1, 1), X_t, Y_t.view(-1, 1)
        ], dim=1).to(dtype=torch.float64)
        
        # Network architecture
        dim = self.hparams.get('dim_layer', 50)
        cfg_density = [(indim, dim, 1, 'relu'), (dim, dim, 1, 'relu')]
        cfg = [(dim, dim, 1, 'relu'), (dim, 1, 1, 'id')]
        num_grid = self.hparams.get('num_grid', 10)
        degree = self.hparams.get('degree', 2)
        knots = [0.33, 0.66]
        
        # Build VCNet model
        model = self.DNet.Vcnet(cfg_density, num_grid, cfg, degree, knots)
        model._initialize_weights()
        model.to(self._device, dtype=torch.float64)
        
        # Hyperparameters (from GIKS paper for IHDP)
        lr = self.hparams.get('lr', 1e-3)
        wd = self.hparams.get('weight_decay', 5e-3)
        num_epochs = self.hparams.get('epoch_total', 200)
        batch_size = self.hparams.get('batch_size', 128)
        gi_lambda = self.hparams.get('gi_lambda', 1e-4)  # GI regularization strength
        gi_linear_delta = self.hparams.get('gi_linear_delta', 0.05)  # neighborhood for GI
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        mse_loss = nn.MSELoss()
        
        # DataLoader
        from torch.utils.data import DataLoader
        dataset = self.Dataset_from_matrix(train_matrix)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Phase 1: Factual pre-training (100 epochs)
        phase1_epochs = num_epochs // 2
        for epoch in range(phase1_epochs):
            model.train()
            for batch_inp, batch_y, batch_ids in train_loader:
                optimizer.zero_grad()
                batch_dosage = batch_inp[:, 0].to(self._device, dtype=torch.float64)
                batch_x = batch_inp[:, 1:].to(self._device, dtype=torch.float64)
                batch_y = batch_y.to(self._device, dtype=torch.float64)
                
                out = model.forward(dosage=batch_dosage, x=batch_x)
                loss = mse_loss(out[1].squeeze(), batch_y.squeeze())
                loss.backward()
                optimizer.step()
        
        # Phase 2: GIKS training with GI augmentation (100 more epochs)
        phase2_epochs = num_epochs - phase1_epochs
        for epoch in range(phase2_epochs):
            model.train()
            for batch_inp, batch_y, batch_ids in train_loader:
                optimizer.zero_grad()
                batch_dosage = batch_inp[:, 0].to(self._device, dtype=torch.float64)
                batch_x = batch_inp[:, 1:].to(self._device, dtype=torch.float64)
                batch_y = batch_y.to(self._device, dtype=torch.float64)
                
                # Factual loss
                out = model.forward(dosage=batch_dosage, x=batch_x)
                factual_loss = mse_loss(out[1].squeeze(), batch_y.squeeze())
                
                # GI loss: perturb dosage slightly, use Taylor expansion
                # Sample small perturbation deltas
                delta = torch.FloatTensor(len(batch_dosage)).uniform_(
                    -gi_linear_delta, gi_linear_delta
                ).to(self._device, dtype=torch.float64)
                
                perturbed_dosage = (batch_dosage - delta).detach().requires_grad_(True)
                perturbed_dosage_clamped = torch.clamp(perturbed_dosage, 0.0, 1.0)
                
                # Get embeddings
                embeddings = model.hidden_features(batch_x)
                hidden = nn.ReLU()(embeddings)
                
                # Forward with perturbed dosage using embeddings
                t_hidden = torch.cat((perturbed_dosage_clamped.unsqueeze(1), hidden.detach()), 1)
                g = model.density_estimator_head(perturbed_dosage_clamped, hidden.detach())
                gi_preds = model.Q(t_hidden)
                
                # Compute gradients w.r.t. dosage
                gi_grads = torch.autograd.grad(
                    gi_preds, perturbed_dosage,
                    grad_outputs=torch.ones_like(gi_preds),
                    create_graph=True
                )[0]
                
                # Taylor expansion: y(t) ≈ y(t-δ) + δ * dy/dt
                taylor_preds = gi_preds.squeeze() + delta * gi_grads
                gi_loss = mse_loss(taylor_preds, batch_y.squeeze())
                
                # Combined loss
                total_loss = factual_loss + gi_lambda * gi_loss
                total_loss.backward()
                optimizer.step()
        
        self.model = model
        
    def predict(self, X, T):
        """
        Predict outcomes for given X and T.
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float64).to(self._device)
            
            if np.isscalar(T):
                T_t = torch.full((X_t.shape[0],), T, dtype=torch.float64).to(self._device)
            else:
                T_t = torch.tensor(T, dtype=torch.float64).to(self._device)
            
            out = self.model.forward(dosage=T_t, x=X_t)
            preds = out[1].squeeze().cpu().numpy()
            
        return preds
