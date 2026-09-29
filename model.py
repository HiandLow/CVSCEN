import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data
import matplotlib.pyplot as plt
import utils

def rbf_kernel(X, sigma=None):
    XX = X.matmul(X.t())
    X_sqnorms = torch.diagonal(XX)
    r = X_sqnorms.unsqueeze(0) - 2 * XX + X_sqnorms.unsqueeze(1)

    if sigma is None:
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
    Kc = K - K.mean(dim=0, keepdim=True) - K.mean(dim=1, keepdim=True) + K.mean()
    hsic = torch.sum(Kc * L) / ((n - 1) ** 2)
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
        self.logits.data[:, 1] = getattr(self, 'init_logit', 1.0)
        self.guidance = getattr(self, 'guidance', 'logit')
        if self.guidance not in ('penalty', 'logit'):
            raise ValueError(f"Unknown guidance: {self.guidance}")
        hsic = self.HSIC_xa
        score = torch.clamp(torch.tanh((hsic - hsic.mean()) / (hsic.std() + 1e-8)), max=0.0).to(self.device)
        weight_corr = getattr(self, 'weight_corr', 0.0)
        if self.guidance == 'penalty':
            self.c_weight = 1.0 - weight_corr * score
            self.c_shift = torch.zeros_like(score)
        else:
            self.c_weight = torch.ones_like(score)
            self.c_shift = weight_corr * score
        self.temp = self.temp_start
        self.use_corr = False
        self.epoch = -1

    def guided_logits(self):
        if not self.use_corr:
            return self.logits
        zeros = torch.zeros_like(self.c_shift)
        return self.logits + torch.stack([zeros, self.c_shift, zeros], dim=1)

    def forward(self, epoch):
        if self.training and epoch > self.epoch:
            self.temp = self.temp_start * (self.temp_end / self.temp_start) ** (epoch / self.epoch_total)
            self.use_corr = self.temp <= 1.0
            self.epoch = epoch

        logits = self.guided_logits()
        if self.training:
            m = F.gumbel_softmax(logits, tau=self.temp, hard=False, dim=1)
        else:
            m = F.one_hot(torch.argmax(logits, dim=-1), num_classes=3).float()

        return m[:, 1], m[:, 2]

class GateLayer(BaseModule):
    def __init__(self, info, hparams):
        super().__init__(info, hparams)

        self.temp_start = 10.0
        self.temp_end = 0.1
        init = getattr(self, 'init_logit', 1.0)
        self.logits_a = torch.nn.Parameter(torch.zeros((self.dim_x, 2), device=self.device))
        self.logits_y = torch.nn.Parameter(torch.zeros((self.dim_x, 2), device=self.device))
        self.logits_a.data[:, 1] = init
        self.logits_y.data[:, 1] = init
        self.c_weight = torch.ones(self.dim_x, device=self.device)
        self.temp = self.temp_start
        self.use_corr = False
        self.epoch = -1

    def gate(self, logits):
        if self.training:
            return F.gumbel_softmax(logits, tau=self.temp, hard=False, dim=1)[:, 1]
        return torch.argmax(logits, dim=-1).float()

    def forward(self, epoch):
        if self.training and epoch > self.epoch:
            self.temp = self.temp_start * (self.temp_end / self.temp_start) ** (epoch / self.epoch_total)
            self.use_corr = self.temp <= 1.0
            self.epoch = epoch

        self.m_a = self.gate(self.logits_a)
        self.m_y = self.gate(self.logits_y)
        m_a = self.m_a.detach()
        return m_a * self.m_y, (1 - m_a) * self.m_y

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
        self.dim_a_expanded = 1 + 2 * self.L_pe

        self.fe_c_shared = torch.nn.Linear(self.dim_x, self.dim_layer)
        self.fe_c_base = ResidualBlock(self.dim_layer)
        self.fe_c_cate = ResidualBlock(self.dim_layer)

        self.fe_p_shared = torch.nn.Linear(self.dim_x, self.dim_layer)
        self.fe_p_base = ResidualBlock(self.dim_layer)
        self.fe_p_cate = ResidualBlock(self.dim_layer)

        self.pr_base = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * self.dim_layer, self.dim_layer),
            torch.nn.SiLU(),
            torch.nn.Linear(self.dim_layer, self.dim_y)
        )

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
        encoding = [a]
        for k in range(self.L_pe):
            encoding.append(torch.sin((2 ** k) * np.pi * a))
            encoding.append(torch.cos((2 ** k) * np.pi * a))
        return torch.cat(encoding, dim=-1)

    def forward(self, x_c, x_p, a):
        a_expanded = self.expand_a(a)

        c_shared = self.fe_c_shared(x_c)
        rep_c_base = self.fe_c_base(c_shared)
        rep_c_cate = self.fe_c_cate(c_shared)

        p_shared = self.fe_p_shared(x_p)
        rep_p_base = self.fe_p_base(p_shared)
        rep_p_cate = self.fe_p_cate(p_shared)

        y_base = self.pr_base(torch.cat([rep_c_base, rep_p_base], dim=-1))

        cate_feat = self.pr_cate_feature(torch.cat([rep_c_cate, rep_p_cate], dim=-1))
        gamma_a = self.ce_gamma(a_expanded)
        beta_a = self.ce_beta(a_expanded)
        y_cate = self.pr_cate_linear(cate_feat * (1 + gamma_a) + beta_a)

        return y_base + y_cate

class MainModel(BaseModule):
    def __init__(self, info, hparams):
        super().__init__(info, hparams)
        self.disable_vs = hparams.get('disable_vs', False)
        self.use_mlp = hparams.get('use_mlp', False)
        self.use_oracle = hparams.get('use_oracle', False)
        self.two_gate = isinstance(self.vsl, GateLayer)

        if self.two_gate:
            self.treat_head = nn.Sequential(
                nn.Linear(self.dim_x, self.dim_layer),
                nn.SiLU(),
                nn.Linear(self.dim_layer, self.dim_layer),
                nn.SiLU(),
                nn.Linear(self.dim_layer, 1)
            )

        if self.use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(self.dim_x * 2 + 1, self.dim_layer),
                nn.ReLU(),
                nn.Linear(self.dim_layer, self.dim_layer),
                nn.ReLU(),
                nn.Linear(self.dim_layer, self.dim_y)
            )

    def predict(self, x_c, x_p, a):
        if self.use_mlp:
            return self.mlp(torch.cat([x_c, x_p, a], dim=-1))
        return self.pol(x_c, x_p, a)

    def forward(self, x, a, epoch):
        m_c, m_p = self.vsl(epoch)

        if self.disable_vs:
            m_c = torch.ones_like(m_c)
            m_p = torch.ones_like(m_p)
        elif self.use_oracle:
            m_c = self.oracle_c.to(x.device)
            m_p = self.oracle_p.to(x.device)

        x_c, x_p = x * m_c, x * m_p
        if self.two_gate:
            self.a_hat = self.treat_head(x * self.vsl.m_a)
        return self.predict(x_c, x_p, a), x_c, x_p, m_c, m_p

    def aux_terms(self, x_p, a, m_c, m_p, coef_loss):
        if self.two_gate:
            aux = F.mse_loss(self.a_hat, a)
            reg = coef_loss[1] * self.vsl.m_y.mean() + coef_loss[2] * self.vsl.m_a.mean()
        else:
            aux = hsic_loss(x_p, a.view(-1, 1))
            reg = coef_loss[1] * (m_c * self.vsl.c_weight).mean() + coef_loss[2] * m_p.mean()
        return aux, reg

def build(info, hparams):
    selector_cls = GateLayer if hparams.get('selector', 'role') == 'two_gate' else VSLayer
    selector = selector_cls(info, hparams)
    predictor_y = POLayer(info, hparams)
    model_main = MainModel(info=dict(info, vsl=selector, pol=predictor_y), hparams=hparams)

    if model_main.use_mlp:
        param_p = list(model_main.mlp.parameters())
    else:
        param_p = list(predictor_y.parameters())
    if model_main.two_gate:
        param_p += list(model_main.treat_head.parameters())

    optimizer_s = torch.optim.Adam(selector.parameters(), lr=hparams["lr_s"])
    optimizer_p = torch.optim.Adam(param_p, lr=hparams["lr_p"], weight_decay=hparams.get("weight_decay", 1e-3))
    scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=100, gamma=0.97)
    scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=100, gamma=0.97)

    if model_main.two_gate:
        coef_loss = [hparams["weight_treat"], hparams["coef_loss_y"], hparams["coef_loss_a"]]
    else:
        coef_loss = [hparams["weight_hsic"], hparams["coef_loss_c"], hparams["coef_loss_p"]]
    return model_main, optimizer_s, optimizer_p, scheduler_s, scheduler_p, coef_loss

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
            loss_hsic_val, loss_reg = model.aux_terms(x_p, a, m_c, m_p, coef_loss)
            anneal_weight = 1.0 if model.vsl.use_corr else 0.0
            loss = loss_y + coef_loss[0] * loss_hsic_val + loss_reg * anneal_weight

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
                loss_val_y.append(F.smooth_l1_loss(y_star_hat, yf, beta=1.0).item())
                loss_val_hsic.append(model.aux_terms(x_p, a, m_c, m_p, coef_loss)[0].item())

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
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./figures/learning_curve.png')
    plt.close()

    return model, history_train_y, history_val_y

def predict_curves(model, loader, t_range=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        if t_range is None:
            all_a = torch.cat([a for _, a, _ in loader]).cpu().numpy()
            t_range = (np.percentile(all_a, 5), np.percentile(all_a, 95))
        t_min, t_max = t_range

        grid_size = 2 ** 6 + 1
        dx = (t_max - t_min) / (grid_size - 1)
        treat_grid = torch.linspace(t_min, t_max, grid_size, device=device)

        preds, xs = [], []
        for x, a, _ in loader:
            x, a = x.to(device), a.to(device)
            _, x_c, x_p, m_c, m_p = model(x, a, epoch=model.epoch_total)
            batch_size = x.shape[0]
            x_c_rep = x_c.repeat_interleave(grid_size, dim=0)
            x_p_rep = x_p.repeat_interleave(grid_size, dim=0)
            a_rep = treat_grid.repeat(batch_size).view(-1, 1)
            pred = model.predict(x_c_rep, x_p_rep, a_rep).view(batch_size, grid_size).cpu().numpy()
            preds.append(pred)
            xs.append(x.cpu().numpy())

    return np.concatenate(preds), np.concatenate(xs), treat_grid.cpu().numpy(), dx, (m_c, m_p)

def evaluate(model, loader_test, coefs, dataset_type='ihdp', t_range=None, x_ref=None):
    pred_grid, x_all, grid_np, dx, (m_c, m_p) = predict_curves(model, loader_test, t_range)
    fact_grid = data.true_curves(x_all, grid_np, dataset_type, [c.cpu() for c in coefs], X_ref=x_ref)

    coef_a, coef_y = coefs[0].to(m_c.device), coefs[1].to(m_c.device)
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

    c_c = m_c.bool().cpu().numpy().astype(int)
    c_p = m_p.bool().cpu().numpy().astype(int)
    c_cp = (m_c.bool() | m_p.bool()).cpu().numpy().astype(int)

    mise = utils.curve_error(pred_grid, fact_grid, dx)
    adrfe = utils.curve_error(pred_grid, fact_grid, dx, adrf_only=True)
    return mise, adrfe, fdr, tpr, [c_c, c_p, c_cp], [pred_grid.mean(axis=0), fact_grid.mean(axis=0), grid_np]
