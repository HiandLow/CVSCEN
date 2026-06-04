import sys
import os
import numpy as np

class VCNetWrapper:
    def __init__(self, num_features, model_name='Vcnet_tr', n_epochs=500, **kwargs):
        import torch
        self.device = torch.device("cpu")
        self.model_name = model_name
        self.n_epochs = n_epochs
        
        # Add original VCNet directory to path to import their models
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            
        vscen_data = sys.modules.pop('data', None)
        from dynamic_net import Vcnet, TR, Drnet
        if vscen_data is not None:
            sys.modules['data'] = vscen_data
        
        self.dim = kwargs.get('dim_layer', 50)
        self.init_lr = kwargs.get('lr', 0.001 if 'Vcnet' in self.model_name else 0.02)
        self.alpha = kwargs.get('alpha', 0.5)
        self.n_epochs = kwargs.get('epoch_total', n_epochs)

        # Default cfg
        cfg_density = [(num_features, self.dim, 1, 'relu'), (self.dim, self.dim, 1, 'relu')]
        num_grid = 10
        cfg = [(self.dim, self.dim, 1, 'relu'), (self.dim, 1, 1, 'id')]
        degree = 2
        knots = [0.33, 0.66]
        
        if self.model_name in ['Vcnet', 'Vcnet_tr']:
            self.model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
        elif self.model_name in ['Drnet', 'Drnet_tr']:
            self.model = Drnet(cfg_density, num_grid, cfg, isenhance=1)
            
        self.model._initialize_weights()
        self.model.to(self.device)
        
        self.isTargetReg = 1 if '_tr' in self.model_name else 0
        if self.isTargetReg:
            tr_knots = list(np.arange(0.05, 1, 0.05))
            tr_degree = 2
            self.TargetReg = TR(tr_degree, tr_knots)
            self.TargetReg._initialize_weights()
            self.TargetReg.to(self.device)
            
    def fit(self, X, T, Y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        self.model.train()
        if self.isTargetReg:
            self.TargetReg.train()
            
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                torch.tensor(T, dtype=torch.float32), 
                                torch.tensor(Y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=471, shuffle=True)
        
        # optimizers
        init_lr = self.init_lr
        alpha = self.alpha
        beta = 1.
        wd = 5e-3
        momentum = 0.9
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=init_lr, momentum=momentum, weight_decay=wd, nesterov=True)
        if self.isTargetReg:
            tr_optimizer = torch.optim.SGD(self.TargetReg.parameters(), lr=0.001, weight_decay=1e-3)
            
        def criterion(out, y, alpha=0.5, epsilon=1e-6):
            return ((out[1].squeeze() - y.squeeze()) ** 2).mean() - alpha * torch.log(out[0] + epsilon).mean()
        def criterion_TR(out, trg, y, beta=1., epsilon=1e-6):
            return beta * ((y.squeeze() - trg.squeeze() / (out[0].squeeze() + epsilon) - out[1].squeeze()) ** 2).mean()

        for epoch in range(self.n_epochs):
            for x, t, y in loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                if self.isTargetReg:
                    optimizer.zero_grad()
                    out = self.model(t, x)
                    trg = self.TargetReg(t)
                    loss = criterion(out, y, alpha=alpha) + criterion_TR(out, trg, y, beta=beta)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    tr_optimizer.zero_grad()
                    out = self.model(t, x)
                    trg = self.TargetReg(t)
                    tr_loss = criterion_TR(out, trg, y, beta=beta)
                    tr_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.TargetReg.parameters(), 1.0)
                    tr_optimizer.step()
                else:
                    optimizer.zero_grad()
                    out = self.model(t, x)
                    loss = criterion(out, y, alpha=alpha)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

    def predict(self, X, T):
        import torch
        self.model.eval()
        if self.isTargetReg:
            self.TargetReg.eval()
            
        if np.isscalar(T):
            T = np.full(X.shape[0], T)
            
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        T_t = torch.tensor(T, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            if self.isTargetReg:
                out = self.model(T_t, X_t)
                trg = self.TargetReg(T_t)
                pred = (trg.squeeze() / (out[0].squeeze() + 1e-6)) + out[1].squeeze()
            else:
                out = self.model(T_t, X_t)
                pred = out[1].squeeze()
                
        return pred.cpu().numpy()
