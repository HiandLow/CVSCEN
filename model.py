import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data
from scipy.integrate import romb
import matplotlib.pyplot as plt
import utils
import copy

def rbf_kernel(X, sigma=1.0):
    XX = X.matmul(X.t())
    X_sqnorms = torch.diagonal(XX)
    r = X_sqnorms.unsqueeze(0) - 2 * XX + X_sqnorms.unsqueeze(1)
    K = torch.exp(-r / (2 * sigma ** 2))
    return K

def hsic_loss(X, Y, sigma_x=1.0, sigma_y=1.0):
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
            self.HSIC[:, 1] = self.HSIC_xa * self.use_corr * self.weight_corr
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

        self.dim_a_expanded = 1 + 2 * self.L_pe  # [a, sin(2^0*pi*a), cos(2^0*pi*a), ..., sin(2^(L_pe-1)*pi*a), cos(2^(L_pe-1)*pi*a)]

        fe_layers = [torch.nn.Linear(self.dim_x, self.dim_layer)]
        for _ in range(getattr(self, 'num_layers', 2)):
            fe_layers.append(ResidualBlock(self.dim_layer))
        self.fe = torch.nn.Sequential(*fe_layers)
        
        self.ce_gamma = torch.nn.Sequential(
            torch.nn.Linear(self.dim_a_expanded, self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_layer, self.dim_layer))   
        
        self.ce_beta = torch.nn.Sequential(
            torch.nn.Linear(self.dim_a_expanded, self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_layer, self.dim_layer)) 

        self.pr = torch.nn.Sequential(
            torch.nn.LayerNorm(self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_layer, self.dim_y))
        
    def expand_a(self, a):
        # NeRF-style positional encoding: [a, sin(2^k * pi * a), cos(2^k * pi * a)] for k=0,...,L_pe-1
        encoding = [a]
        for k in range(self.L_pe):
            encoding.append(torch.sin((2 ** k) * np.pi * a))
            encoding.append(torch.cos((2 ** k) * np.pi * a))
        return torch.cat(encoding, dim=-1)
    
    def forward(self, uv, a):
        a_expanded = self.expand_a(a)
        y_star_hat = self.pr(self.fe(uv) * (1 + self.ce_gamma(a_expanded)) + self.ce_beta(a_expanded))

        return y_star_hat
    
class MainModel(BaseModule):
    def __init__(self, info, hparams):
        super().__init__(info, hparams)

    def forward(self, x, a, epoch):
        m_u, m_v = self.vsl(epoch)
        u, v = x * m_u, x * m_v
        y_star_hat = self.pol(u + v, a)

        return y_star_hat, u, m_u, m_v
    
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
            y_star_hat, u, m_u, m_v = model(x, a, epoch)

            loss_y = F.smooth_l1_loss(y_star_hat, yf, beta=1.0)
            
            # Use HSIC to penalize dependency between Outcome-Predictors (u) and Treatment (a)
            # a is safely reshaped to 2D (batch_size, 1) for HSIC
            loss_hsic_val = hsic_loss(u, a.view(-1, 1), sigma_x=1.0, sigma_y=1.0)
            
            loss = coef_loss[0] * loss_y + coef_loss[1] * loss_hsic_val + coef_loss[2] * m_u.mean() + coef_loss[3] * m_v.mean()


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
                y_star_hat, u, m_u, m_v = model(x, a, epoch)
                
                loss_y = F.smooth_l1_loss(y_star_hat, yf, beta=1.0)
                loss_hsic_val = hsic_loss(u, a.view(-1, 1), sigma_x=1.0, sigma_y=1.0)
                loss = coef_loss[0] * loss_y + coef_loss[1] * loss_hsic_val + coef_loss[2] * m_u.mean() + coef_loss[3] * m_v.mean()

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
    plt.savefig('./result/learning_curve.png')

    return model

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
        total_mise = 0.0 
        total_samples = 0
        c_u = np.zeros(model.vsl.dim_x, dtype=int)
        c_v = np.zeros(model.vsl.dim_x, dtype=int)
        c_uv = np.zeros(model.vsl.dim_x, dtype=int)
        
        all_pred_curves = []
        all_fact_curves = []
        
        for x, a, yf in loader_test:
            x, a, yf = x.to(device), a.to(device), yf.to(device)
            y_star_hat, u, m_u, m_v = model(x, a, epoch=model.epoch_total)
            uv = (x * m_u + x * m_v)   

            grid_size = 2 ** 6 + 1
            dx = 1. / (grid_size - 1)
            treat_grid = torch.linspace(np.finfo(float).eps, 1, grid_size, device=device)

            pred_grid = torch.cat([model.pol(uv, treat.expand_as(a)) for treat in treat_grid], dim=1)
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
            utils.FDR(m_u, coef_a, coef_y, is_confounder=True), 
            utils.FDR(m_v, coef_a, coef_y, is_confounder=False), 
            utils.FDR(m_u + m_v, coef_a + coef_y, coef_y + coef_y, is_confounder=True)
        ]

        tpr = [
            utils.TPR(m_u, coef_a, coef_y, is_confounder=True), 
            utils.TPR(m_v, coef_a, coef_y, is_confounder=False), 
            utils.TPR(m_u + m_v, coef_a + coef_y, coef_y + coef_y, is_confounder=True)
        ]
        
        if m_v.dim() > 1:
            c_v += m_v.bool().sum(dim=0).cpu().numpy().astype(int)
            c_u += m_u.bool().sum(dim=0).cpu().numpy().astype(int)
            c_uv += (m_u.bool() | m_v.bool()).sum(dim=0).cpu().numpy().astype(int)
        else:
            c_v += m_v.bool().cpu().numpy().astype(int)
            c_u += m_u.bool().cpu().numpy().astype(int)
            c_uv += (m_u.bool() | m_v.bool()).cpu().numpy().astype(int)

        avg_pred = torch.cat(all_pred_curves, dim=0).mean(dim=0).numpy()
        avg_fact = torch.cat(all_fact_curves, dim=0).mean(dim=0).numpy()
        
        adrfe = romb((avg_fact - avg_pred) ** 2, dx=dx)
        mise = total_mise / total_samples

        t_axis = treat_grid.cpu().numpy()


    return np.sqrt(mise), np.sqrt(adrfe), fdr, tpr, [c_u, c_v, c_uv], [avg_pred, avg_fact, t_axis]