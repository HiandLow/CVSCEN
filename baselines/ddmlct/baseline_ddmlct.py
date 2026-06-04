import sys
import os
import numpy as np

class DDMLCTWrapper:
    def __init__(self, num_features, **kwargs):
        self.num_features = num_features
        self.hparams = kwargs
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:

            sys.path.insert(0, current_dir)
            
        import Supplement.estimation as est
        import Supplement.models as mods
        
        self.est = est
        self.mods = mods
        
    def fit(self, X, T, Y):
        import pandas as pd
        import torch
        
        # We need to construct pandas DataFrames as expected by their code
        # Their code assumes T is pandas Series, Y is pandas Series, X is DataFrame
        X_df = pd.DataFrame(X).astype('float64')
        T_series = pd.Series(T).astype('float64')
        Y_series = pd.Series(Y).astype('float64')
        
        # Use their Neural Network models
        # model1 is used for estimating the conditional expectation of Y (Outcome Model)
        # model2 is used for estimating the conditional density/GPS (Treatment Model)
        # We use NeuralNet1k_emp_app and NeuralNet2_emp_app
        lr1 = self.hparams.get('lr', 0.05)
        lr2 = self.hparams.get('lr_s', 0.01)
        epochs = self.hparams.get('epoch_total', 300)
        
        model_knn1 = self.mods.NeuralNet1k_emp_app(k=self.num_features, lr=lr1, momentum=0.9, epochs=epochs, weight_decay=0.01)
        model_knn2 = self.mods.NeuralNet2_emp_app(k=self.num_features, lr=lr2, momentum=0.9, epochs=epochs, weight_decay=0.01)
        
        # Grid for dose-response curve
        grid_size = 2**6 + 1
        t_min, t_max = np.percentile(T, 5), np.percentile(T, 95)
        self.t_list = np.linspace(t_min, t_max, grid_size)
        
        # 1. First round estimation with rule of thumb bandwidth
        L = 5
        h = np.std(T) * 3 * (len(Y)**(-0.2))
        u = 0.5
        
        # Fit model with h
        model1 = self.est.NN_DDMLCT(model_knn1, model_knn2)
        model1.fit(X_df, T_series, Y_series, self.t_list, L=L, h=h, basis=False, standardize=True)
        
        # Fit model with h * u
        # Need to re-instantiate NN components so they train fresh, though DDMLCT fit resets them anyway
        model_knn1_b = self.mods.NeuralNet1k_emp_app(k=self.num_features, lr=lr1, momentum=0.9, epochs=epochs, weight_decay=0.01)
        model_knn2_b = self.mods.NeuralNet2_emp_app(k=self.num_features, lr=lr2, momentum=0.9, epochs=epochs, weight_decay=0.01)
        model2 = self.est.NN_DDMLCT(model_knn1_b, model_knn2_b)
        model2.fit(X_df, T_series, Y_series, self.t_list, L=L, h=h*u, basis=False, standardize=True)
        
        # 2. Compute h_star (optimal bandwidth)
        Bt = (model1.beta - model2.beta) / ((model1.h**2) * (1 - (u**2)))
        # Add small epsilon to avoid division by zero
        h_star = np.mean(((model2.Vt / (4 * (Bt**2) + 1e-8))**0.2) * (model1.n**-0.2))
        
        # 3. Final estimation with h_star
        h_final = 0.8 * h_star # Paper recommends scaling slightly
        
        model_knn1_final = self.mods.NeuralNet1k_emp_app(k=self.num_features, lr=lr1, momentum=0.9, epochs=epochs, weight_decay=0.01)
        model_knn2_final = self.mods.NeuralNet2_emp_app(k=self.num_features, lr=lr2, momentum=0.9, epochs=epochs, weight_decay=0.01)
        self.model = self.est.NN_DDMLCT(model_knn1_final, model_knn2_final)
        self.model.fit(X_df, T_series, Y_series, self.t_list, L=L, h=h_final, basis=False, standardize=True)
        
        # Store mapping of t -> beta(t)
        self.beta_map = {t: b for t, b in zip(self.t_list, self.model.beta)}
        
    def predict(self, X, T):
        # The evaluation harness expects predictions for each individual X.
        # Since this method natively estimates the Average Dose-Response Function,
        # we return the global beta(t) for all individuals.
        n = X.shape[0]
        if np.isscalar(T):
            # T is a scalar
            idx = np.argmin(np.abs(self.t_list - T))
            val = self.beta_map[self.t_list[idx]]
            return np.full(n, val)
        else:
            # T is an array
            res = np.zeros(n)
            for i in range(n):
                idx = np.argmin(np.abs(self.t_list - T[i]))
                res[i] = self.beta_map[self.t_list[idx]]
            return res
