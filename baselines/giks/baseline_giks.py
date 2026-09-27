import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GIKSWrapper:
    """
    Wrapper for the GIKS (Gradient Interpolation and Kernel Smoothing) model
    from Nagalapatti et al., AAAI 2024.

    Follows run_GIKS in continuous/main_helper.py with its default config
    (GP_UNF_T=True, BTM_K_VAR=SM, GP_KERNEL=cosine, CTR_SAMPLING_DIST=uniform):
    1. Factual pre-training of a VCNet backbone
    2. Factual loss + gp_lambda * (GI loss for near counterfactual dosages
       + kernel-smoothing (GP) loss for far ones)
    The best epoch by validation RMSE on a held-out slice of the training data
    is restored, as the original's early stopping does (val_pc=0.3).
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
        import constants as giks_constants

        self.DNet = DNet
        self.C = giks_constants

        # Restore project dir to sys.path
        if project_dir not in sys.path:
            if path_was_first:
                sys.path.insert(0, project_dir)
            else:
                sys.path.append(project_dir)

        # Restore our utils module (but keep GIKS utils cached too under different keys)
        for key, mod in saved_modules.items():
            sys.modules[key] = mod

    # ---- Ported from continuous/main_helper.py: sample_linear_delta, GI_reg_loss,
    # ---- sample_far_dosages (uniform), GP_loss (BTM_K_VAR=SM), GP_unf_loss

    def _sample_linear_delta(self, dosage, num_samples, linear_delta):
        delta_samples = torch.empty(len(dosage), num_samples, device=self._device,
                                    dtype=torch.float64).uniform_(-linear_delta, linear_delta)
        dosage_delta = dosage.view(-1, 1) - delta_samples
        delta_samples[dosage_delta < 0] = 0
        delta_samples[dosage_delta > 1] = 1
        return delta_samples

    def _gi_loss(self, model, batch_dosage, batch_dosage_grad, batch_xemb, batch_y, linear_delta, num_explore=1):
        delta_samples = self._sample_linear_delta(batch_dosage, num_explore, linear_delta).view(-1)
        batch_dosage_gi = torch.repeat_interleave(batch_dosage_grad, repeats=num_explore)
        batch_dosage_delta = batch_dosage_gi - delta_samples
        x_emb = torch.repeat_interleave(batch_xemb, repeats=num_explore, dim=0)
        gi_tgt_y = torch.repeat_interleave(batch_y, repeats=num_explore)
        gi_ypreds = model.forward_with_emb(dosage=batch_dosage_delta, x_emb=x_emb)[1]
        gi_grads = torch.autograd.grad(gi_ypreds, batch_dosage_delta,
                                       grad_outputs=torch.ones_like(gi_ypreds), create_graph=True)[0]
        ypreds_taylor = gi_ypreds.view(-1) + delta_samples * gi_grads
        return nn.MSELoss()(gi_tgt_y, ypreds_taylor)

    def _sample_far_dosages(self, dosage, linear_delta):
        # Uniform over [0, 1] outside the +/- linear_delta window around the factual dosage
        sampled = []
        for d in dosage.tolist():
            lo, hi = min(max(0, d - linear_delta), 1), min(max(0, d + linear_delta), 1)
            s = np.random.uniform(0, 1 - (hi - lo))
            if s > hi:  # same shift rule as the original
                s += hi - lo
            sampled.append(s)
        return torch.tensor(sampled, device=self._device, dtype=torch.float64)

    def _gp_loss(self, model, batch_dosage_f, batch_emb, trn_dosages, trn_ys, trn_embs,
                 gi_linear_delta, gp_linear_delta, sm_temp):
        far_dosage_CF = self._sample_far_dosages(batch_dosage_f, gi_linear_delta)
        diff = torch.abs(far_dosage_CF.view(-1, 1) - trn_dosages.view(1, -1))

        means, variances, keep = [], [], []
        with torch.no_grad():
            for i in range(len(far_dosage_CF)):
                nnd_ids = torch.where(diff[i] < gp_linear_delta)[0]
                if len(nnd_ids) == 0:
                    # The original produces NaN here; skip such samples instead
                    continue
                nnd_emb, nnd_y = trn_embs[nnd_ids], trn_ys[nnd_ids]
                ymean = torch.mean(nnd_y)
                gp = self.DNet.GP_NN()
                gp.forward(Z_f=nnd_emb, **{self.C.GP_KERNEL: self.C.COSINE_KERNEL})
                mean_w = gp.mean_w(nnd_y - ymean)
                fx = F.normalize(batch_emb[i].detach().clone(), p=2, dim=0)
                means.append(torch.sum(mean_w.T * fx, axis=-1).squeeze() + ymean)
                variances.append((fx.view(1, -1) @ gp.ker_inv @ fx.view(-1, 1)).squeeze())
                keep.append(i)
        if not keep:
            return torch.zeros((), device=self._device, dtype=torch.float64)

        keep = torch.tensor(keep, device=self._device)
        means, variances = torch.stack(means).view(-1), torch.stack(variances).view(-1)
        out_cf = model.forward_with_emb(dosage=far_dosage_CF[keep], x_emb=batch_emb[keep])
        # Softmax over negated GP variances: low-variance targets get more weight
        weight = F.softmax(-variances / sm_temp, dim=0)
        return torch.sum(weight * (out_cf[1].view(-1) - means) ** 2)

    def _gp_unf_loss(self, model, batch_dosage_f, batch_dosage_grad, batch_emb, batch_y,
                     trn_dosages, trn_ys, trn_embs, gi_linear_delta, gp_linear_delta, sm_temp):
        batch_dosage_CF = torch.rand(len(batch_dosage_f), device=self._device, dtype=torch.float64)
        near = torch.where(torch.abs(batch_dosage_f - batch_dosage_CF) < gi_linear_delta)[0]
        far = torch.where(torch.abs(batch_dosage_f - batch_dosage_CF) >= gi_linear_delta)[0]
        loss = torch.zeros((), device=self._device, dtype=torch.float64)
        if len(near) > 0:
            loss = loss + self._gi_loss(model, batch_dosage_f[near], batch_dosage_grad[near],
                                        batch_emb[near], batch_y[near], gi_linear_delta, num_explore=1)
        if len(far) > 0:
            loss = loss + self._gp_loss(model, batch_dosage_f[far], batch_emb[far], trn_dosages, trn_ys,
                                        trn_embs, gi_linear_delta, gp_linear_delta, sm_temp)
        return loss

    def fit(self, X, T, Y):
        """
        X: numpy array (n, d) - covariates
        T: numpy array (n,) - treatments in [0,1]
        Y: numpy array (n,) - outcomes
        """
        dev, f64 = self._device, torch.float64

        # Hold out a validation slice for best-epoch selection (original: val_pc=0.3)
        val_frac = self.hparams.get('val_frac', 0.3)
        perm = np.random.permutation(len(X))
        n_val = int(np.floor(len(X) * val_frac))
        val_idx, trn_idx = perm[:n_val], perm[n_val:]

        trn_xs = torch.tensor(X[trn_idx], dtype=f64, device=dev)
        trn_dosages = torch.tensor(T[trn_idx], dtype=f64, device=dev)
        trn_ys = torch.tensor(Y[trn_idx], dtype=f64, device=dev)
        val_xs = torch.tensor(X[val_idx], dtype=f64, device=dev)
        val_dosages = torch.tensor(T[val_idx], dtype=f64, device=dev)
        val_ys = torch.tensor(Y[val_idx], dtype=f64, device=dev)

        dim = self.hparams.get('dim_layer', 50)
        cfg_density = [(self.num_features, dim, 1, 'relu'), (dim, dim, 1, 'relu')]
        cfg = [(dim, dim, 1, 'relu'), (dim, 1, 1, 'id')]
        model = self.DNet.Vcnet(cfg_density, self.hparams.get('num_grid', 10), cfg,
                                self.hparams.get('degree', 2), [0.33, 0.66])
        model._initialize_weights()
        model.to(dev, dtype=f64)

        # Defaults from continuous/config.py GIKS_ARGS
        lr = self.hparams.get('lr', 1e-2)
        wd = self.hparams.get('weight_decay', 5e-3)
        num_epochs = self.hparams.get('epoch_total', 400)
        batch_size = self.hparams.get('batch_size', 128)
        gp_lambda = self.hparams.get('gp_lambda', 1e-1)
        gi_linear_delta = self.hparams.get('gi_linear_delta', 0.05)
        gp_linear_delta = self.hparams.get('gp_linear_delta', 0.1)
        sm_temp = self.hparams.get('sm_temp', 0.5)
        # Original: factual pre-training, then GI+KS; the split is kept proportional to epoch_total
        pretrain_epochs = int(num_epochs * self.hparams.get('pretrain_frac', 0.5))

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        return_emb = {self.C.RETURN_EMB: True}

        best_val, best_state = float('inf'), None
        n_trn = len(trn_xs)
        for epoch in range(num_epochs):
            model.train()
            order = torch.randperm(n_trn, device=dev)
            for start in range(0, n_trn, batch_size):
                batch_ids = order[start:start + batch_size]
                batch_dosage_grad = trn_dosages[batch_ids].clone().requires_grad_(True)
                batch_dosage = batch_dosage_grad.detach().clone()
                batch_y = trn_ys[batch_ids]

                optimizer.zero_grad()
                # As in the original, embeddings for the whole training set are recomputed every step
                trn_out = model.forward(dosage=trn_dosages, x=trn_xs, **return_emb)
                trn_embs, trn_y_preds = trn_out[2], trn_out[1]
                batch_emb = trn_embs[batch_ids]

                loss = nn.MSELoss()(trn_y_preds[batch_ids].view(-1), batch_y)
                if epoch >= pretrain_epochs:
                    loss = loss + gp_lambda * self._gp_unf_loss(
                        model, batch_dosage, batch_dosage_grad, batch_emb, batch_y,
                        trn_dosages, trn_ys, trn_embs.detach(), gi_linear_delta, gp_linear_delta, sm_temp)
                loss.backward()
                optimizer.step()

            if n_val > 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model.forward(dosage=val_dosages, x=val_xs)[1].view(-1)
                    val_rmse = torch.sqrt(((val_pred - val_ys) ** 2).mean()).item()
                if val_rmse < best_val:
                    best_val, best_state = val_rmse, copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)
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
