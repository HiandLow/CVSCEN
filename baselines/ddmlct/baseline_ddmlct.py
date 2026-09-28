import sys
import os
import contextlib
import numpy as np
import torch

@contextlib.contextmanager
def _float64_default():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

class DDMLCTWrapper:
    def __init__(self, num_features, **kwargs):
        self.num_features = num_features
        self.hparams = kwargs
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        with _float64_default():
            import Supplement.estimation as est
            import Supplement.models as mods

        self.est = est
        self.mods = mods

    def fit(self, X, T, Y):
        with _float64_default():
            self._fit(X, T, Y)

    def _fit(self, X, T, Y):
        import pandas as pd

        X_df = pd.DataFrame(X).astype('float64')
        T_series = pd.Series(T).astype('float64')
        Y_series = pd.Series(Y).astype('float64')

        lr1 = self.hparams.get('lr', 0.05)
        lr2 = self.hparams.get('lr_s', 0.01)
        dim_layer = self.hparams.get('dim_layer', 64)
        epochs = self.hparams.get('epoch_total', 300)
        wd = self.hparams.get('weight_decay', 0.01)
        wd_s = self.hparams.get('weight_decay_s', wd)

        def nets():
            return (self.mods.NeuralNet1k_emp_app(k=self.num_features, dim_layer=dim_layer, lr=lr1, momentum=0.9, epochs=epochs, weight_decay=wd),
                    self.mods.NeuralNet2_emp_app(k=self.num_features, dim_layer=dim_layer, lr=lr2, momentum=0.9, epochs=epochs, weight_decay=wd_s))

        grid_size = 2**6 + 1
        t_min, t_max = np.percentile(T, 5), np.percentile(T, 95)
        self.t_list = np.linspace(t_min, t_max, grid_size)

        L = self.hparams.get('L', 5)
        h = np.std(T) * 3 * (len(Y)**(-0.2))
        u = 0.5

        model1 = self.est.NN_DDMLCT(*nets())
        model1.fit(X_df, T_series, Y_series, self.t_list, L=L, h=h, basis=False, standardize=True)

        model2 = self.est.NN_DDMLCT(*nets())
        model2.fit(X_df, T_series, Y_series, self.t_list, L=L, h=h*u, basis=False, standardize=True)

        Bt = (model1.beta - model2.beta) / ((model1.h**2) * (1 - (u**2)))
        h_star = np.mean(((model2.Vt / (4 * (Bt**2) + 1e-8))**0.2) * (model1.n**-0.2))
        h_final = 0.8 * h_star

        self.model = self.est.NN_DDMLCT(*nets())
        self.model.fit(X_df, T_series, Y_series, self.t_list, L=L, h=h_final, basis=False, standardize=True)

        self.beta = np.asarray(self.model.beta, dtype=np.float64)

    def predict(self, X, T):
        n = X.shape[0]
        if np.isscalar(T):
            return np.full(n, self.beta[np.argmin(np.abs(self.t_list - T))])
        idx = np.abs(self.t_list[None, :] - np.asarray(T).reshape(-1, 1)).argmin(axis=1)
        return self.beta[idx]
