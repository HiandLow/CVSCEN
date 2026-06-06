import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data
from scipy.integrate import romb
import matplotlib.pyplot as plt
import utils
import copy

def rbf_kernel(X, sigma=None):
    XX = X.matmul(X.t())
    X_sqnorms = torch.diagonal(XX)
    r = X_sqnorms.unsqueeze(0) - 2 * XX + X_sqnorms.unsqueeze(1)
    
    if sigma is None:
        # Median heuristic to prevent kernel collapse in high dimensions
        r_flat = r.view(-1)
        mask = r_flat > 1e-8
        if mask.sum() > 0:
            sigma = torch.sqrt(torch.median(r_flat[mask])).item()
        else:
            sigma = 1.0
            
    K = torch.exp(-r / (2 * sigma ** 2))
    return K

def hsic_loss(X, Y, sigma_x=None, sigma_y=None):
    K = rbf_kernel(X, sigma_x)
    L = rbf_kernel(Y, sigma_y)
    n = K.size(0)
    H = torch.eye(n, device=X.device) - (1.0 / n) * torch.ones((n, n), device=X.device)
    Kc = H.matmul(K).matmul(H)
    hsic = torch.trace(Kc.matmul(L)) / ((n - 1) ** 2)
    return hsic

class BaseModule(nn.Module):
    def __init__(self, info, hparams):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for key, value in (info | hparams).items():
            setattr(self, key, value)

class VSLayer(BaseModule):
    def __init__(self, info, hparams):
        super().__init__(info, hparams)

        self.temp_start = 10.0
        self.temp_end = 0.1
        self.logits = torch.nn.Parameter(torch.zeros((self.dim_x, 3), device=self.device))
        self.HSIC_xa = torch.tanh(1.0 * (self.HSIC_xa - self.HSIC_xa.mean()) / self.HSIC_xa.std()).to(self.device)
        self.HSIC = torch.zeros_like(self.logits, device = self.device)
        self.temp = self.temp_start
        self.use_corr = False
        self.epoch = -1

    def forward(self, epoch):
        if self.training and epoch > self.epoch:
            self.temp = self.temp_start * (self.temp_end / self.temp_start) ** (epoch / self.epoch_total)
            self.use_corr = self.temp <= 1.0
            self.HSIC[:, 1] = torch.clamp(self.HSIC_xa, max=0.0) * self.use_corr * self.weight_corr
            self.epoch = epoch

        if self.training:
            m = F.gumbel_softmax(self.logits + self.HSIC,\
                             tau=self.temp, hard=False, dim=1)
            
        else:
            m = torch.argmax(self.logits + self.HSIC, dim=-1)
            m = F.one_hot(m, num_classes=3).float()

        return m[:, 1], m[:, 2]
    
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        return x + self.block(x)

class POLayer(BaseModule):
    def __init__(self, info, hparams):
        super().__init__(info, hparams)
        self.L_pe = 3

        self.dim_a_expanded = 1 + 2 * self.L_pe  # [a, sin(2^0*pi*a), cos(2^0*pi*a), ..., sin(2^(L_pe-1)*pi*a), cos(2^(L_pe-1)*pi*a)]

        # 1. Encoders (Feature Extractors)
        # Shared C Trunk
        self.fe_c_shared = torch.nn.Linear(self.dim_x, self.dim_layer)
        # C Adapters
        self.fe_c_base = ResidualBlock(self.dim_layer)
        self.fe_c_cate = ResidualBlock(self.dim_layer)
        
        # P Encoders
        self.fe_p_shared = torch.nn.Linear(self.dim_x, self.dim_layer)
        self.fe_p_base = ResidualBlock(self.dim_layer)
        self.fe_p_cate = ResidualBlock(self.dim_layer)

        # 2. Base Stream
        self.pr_base = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * self.dim_layer, self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_layer, self.dim_y)
        )

        # 3. CATE Stream
        self.pr_cate_feature = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * self.dim_layer, self.dim_layer)
        )

        self.ce_gamma = torch.nn.Sequential(
            torch.nn.Linear(self.dim_a_expanded, self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_layer, self.dim_layer)
        )   
        
        self.ce_beta = torch.nn.Sequential(
            torch.nn.Linear(self.dim_a_expanded, self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_layer, self.dim_layer)
        ) 

        self.pr_cate_linear = torch.nn.Linear(self.dim_layer, self.dim_y)
        
    def expand_a(self, a):
        # NeRF-style positional encoding: [a, sin(2^k * pi * a), cos(2^k * pi * a)] for k=0,...,L_pe-1
        encoding = [a]
        for k in range(self.L_pe):
            encoding.append(torch.sin((2 ** k) * np.pi * a))
            encoding.append(torch.cos((2 ** k) * np.pi * a))
        return torch.cat(encoding, dim=-1)
    
    def forward(self, x_c, x_p, a):
        a_expanded = self.expand_a(a)
        
        # 1. Extract Features
        c_shared = self.fe_c_shared(x_c)
        rep_c_base = self.fe_c_base(c_shared)
        rep_c_cate = self.fe_c_cate(c_shared)
        
        p_shared = self.fe_p_shared(x_p)
        rep_p_base = self.fe_p_base(p_shared)
        rep_p_cate = self.fe_p_cate(p_shared)
        
        # 2. Base Stream (No Treatment A)
        cat_base = torch.cat([rep_c_base, rep_p_base], dim=-1)
        y_base = self.pr_base(cat_base)

        # 3. CATE Stream (C and P interaction, linearly modulated by A)
        cat_cate = torch.cat([rep_c_cate, rep_p_cate], dim=-1)
        cate_feat = self.pr_cate_feature(cat_cate)
        
        gamma_a = self.ce_gamma(a_expanded) # Removed tanh constraint for expressivity
        beta_a = self.ce_beta(a_expanded)
        
        # Linear/Multiplicative Modulation (No non-linear sharing between P and A)
        phi_cate = cate_feat * (1 + gamma_a) + beta_a
        y_cate = self.pr_cate_linear(phi_cate)
        
        # 4. Final Output
        y_star_hat = y_base + y_cate
        return y_star_hat
    
class MainModel(BaseModule):
    def __init__(self, info, hparams):
        super().__init__(info, hparams)

    def forward(self, x, a, epoch):
        m_c, m_p = self.vsl(epoch)
        x_c, x_p = x * m_c, x * m_p
        y_star_hat = self.pol(x_c, x_p, a)

        return y_star_hat, x_c, x_p, m_c, m_p
    
def train_model(model, optimizer_s, optimizer_p, scheduler_s, scheduler_p, loader_train, loader_val, coef_loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    history_train_y = []
    history_val_y = []

    for epoch in range(model.epoch_total):
        loss_train_y = []
        loss_train_hsic = []

        loss_val_y = []
        loss_val_hsic = []

        model.train()
        for x, a, yf in loader_train:
            optimizer_s.zero_grad()
            optimizer_p.zero_grad()
            x, a, yf = x.to(device), a.to(device), yf.to(device)
            y_star_hat, x_c, x_p, m_c, m_p = model(x, a, epoch)

            loss_y = F.smooth_l1_loss(y_star_hat, yf, beta=1.0)
            
            # Use HSIC to penalize dependency between Prognostic predictors (x_p) and Treatment (a)
            # a is safely reshaped to 2D (batch_size, 1) for HSIC
            loss_hsic_val = hsic_loss(x_p, a.view(-1, 1), sigma_x=None, sigma_y=None)
            
            # Method C (Two-Phase Step Annealing triggered by use_corr)
            anneal_weight = 1.0 if getattr(model.vsl, 'use_corr', False) else 0.0
            
            loss = loss_y + coef_loss[0] * loss_hsic_val + (coef_loss[1] * m_c.mean() + coef_loss[2] * m_p.mean()) * anneal_weight


            loss.backward()
            optimizer_s.step()
            optimizer_p.step()

            loss_train_y.append(loss_y.item())
            loss_train_hsic.append(loss_hsic_val.item())

        scheduler_s.step()
        scheduler_p.step()

        model.eval()
        with torch.no_grad():
            for x, a, yf in loader_val:
                x, a, yf = x.to(device), a.to(device), yf.to(device)
                y_star_hat, x_c, x_p, m_c, m_p = model(x, a, epoch)
                
                loss_y = F.smooth_l1_loss(y_star_hat, yf, beta=1.0)
                loss_hsic_val = hsic_loss(x_p, a.view(-1, 1), sigma_x=None, sigma_y=None)
                
                anneal_weight = 1.0 if getattr(model.vsl, 'use_corr', False) else 0.0
                
                loss = loss_y + coef_loss[0] * loss_hsic_val + (coef_loss[1] * m_c.mean() + coef_loss[2] * m_p.mean()) * anneal_weight

                loss_val_y.append(loss_y.item())
                loss_val_hsic.append(loss_hsic_val.item())

        history_train_y.append(sum(loss_train_y) / len(loss_train_y))
        history_val_y.append(sum(loss_val_y) / len(loss_val_y))

        if epoch % 100 == 0 or epoch == model.epoch_total - 1:
            print(  f"Epoch {epoch:03d}/{model.epoch_total:03d} Done,\n"
                    f"Train: Y_Loss: {history_train_y[-1]:.4f} | "
                    f"HSIC: {sum(loss_train_hsic) / len(loss_train_hsic):.4f}")
            print(  f"Valid: Y_Loss: {history_val_y[-1]:.4f} | "
                    f"HSIC: {sum(loss_val_hsic) / len(loss_val_hsic):.4f} | ")

    epochs_range = range(1, len(history_train_y) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_y, label='Train Y_Loss', color='blue')
    plt.plot(epochs_range, history_val_y, label='Valid Y_Loss', color='red')
    plt.title('Y_Loss over Epochs (Check Overfitting here)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/learning_curve.png')
    plt.close()

    return model, history_train_y, history_val_y

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import romb

def evaluate(model, loader_test, coefs, dataset_type='ihdp', step=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    coef_a, coef_y = coefs[0].to(device), coefs[1].to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        all_a = torch.cat([a for _, a, _ in loader_test]).cpu().numpy()
        t_min, t_max = np.percentile(all_a, 5), np.percentile(all_a, 95)

        total_mise = 0.0 
        total_samples = 0
        c_c = np.zeros(model.vsl.dim_x, dtype=int)
        c_p = np.zeros(model.vsl.dim_x, dtype=int)
        c_cp = np.zeros(model.vsl.dim_x, dtype=int)
        
        all_pred_curves = []
        all_fact_curves = []
        
        for x, a, yf in loader_test:
            x, a, yf = x.to(device), a.to(device), yf.to(device)
            y_star_hat, x_c, x_p, m_c, m_p = model(x, a, epoch=model.epoch_total)
            uv = (x * m_c + x * m_p)   

            grid_size = 2 ** 6 + 1
            dx = (t_max - t_min) / (grid_size - 1)
            treat_grid = torch.linspace(t_min, t_max, grid_size, device=device)

            pred_grid = torch.cat([model.pol(x_c, x_p, treat.expand_as(a)) for treat in treat_grid], dim=1)
            if dataset_type == 'ihdp':
                fact_grid = torch.cat([torch.tensor(data.get_effect_ihdp(uv.cpu().numpy(), treat.item(), coefs), dtype=torch.float32).unsqueeze(1).to(device) for treat in treat_grid], dim=1)
            else:
                fact_grid = torch.cat([torch.tensor(data.get_effect_synt(uv.cpu().numpy(), treat.item(), coefs), dtype=torch.float32).unsqueeze(1).to(device) for treat in treat_grid], dim=1)

            diff_sq = (fact_grid - pred_grid) ** 2
            batch_size = uv.shape[0]
            total_mise += np.sum([romb(diff_sq[idx].cpu().numpy(), dx=dx) for idx in range(batch_size)])
            total_samples += batch_size

            all_pred_curves.append(pred_grid.cpu())
            all_fact_curves.append(fact_grid.cpu())

        fdr = [
            utils.FDR(m_c, coef_a, coef_y, is_confounder=True), 
            utils.FDR(m_p, coef_a, coef_y, is_confounder=False), 
            utils.FDR(m_c + m_p, coef_a + coef_y, coef_y + coef_y, is_confounder=True)
        ]

        tpr = [
            utils.TPR(m_c, coef_a, coef_y, is_confounder=True), 
            utils.TPR(m_p, coef_a, coef_y, is_confounder=False), 
            utils.TPR(m_c + m_p, coef_a + coef_y, coef_y + coef_y, is_confounder=True)
        ]
        
        if m_p.dim() > 1:
            c_p += m_p.bool().sum(dim=0).cpu().numpy().astype(int)
            c_c += m_c.bool().sum(dim=0).cpu().numpy().astype(int)
            c_cp += (m_c.bool() | m_p.bool()).sum(dim=0).cpu().numpy().astype(int)
        else:
            c_p += m_p.bool().cpu().numpy().astype(int)
            c_c += m_c.bool().cpu().numpy().astype(int)
            c_cp += (m_c.bool() | m_p.bool()).cpu().numpy().astype(int)

        avg_pred = torch.cat(all_pred_curves, dim=0).mean(dim=0).numpy()
        avg_fact = torch.cat(all_fact_curves, dim=0).mean(dim=0).numpy()
        
        adrfe = romb((avg_fact - avg_pred) ** 2, dx=dx)
        mise = total_mise / total_samples

        t_axis = treat_grid.cpu().numpy()


    return np.sqrt(mise), np.sqrt(adrfe), fdr, tpr, [c_c, c_p, c_cp], [avg_pred, avg_fact, t_axis]