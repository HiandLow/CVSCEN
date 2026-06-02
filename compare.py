# Lazy imports are used to avoid dependency issues when running in different environments
import numpy as np
import pickle
from scipy.integrate import romb
import data
import utils
import matplotlib.pyplot as plt
import os
import argparse
class LDML:
    def __init__(self):
        from econml.dml import LinearDML
        from sklearn.linear_model import LassoCV
        lasso_model = LassoCV(max_iter=5000, tol=1e-2, eps=1e-2, n_jobs=-1)
        self.model = LinearDML(model_y=lasso_model,
                                model_t=lasso_model,
                                discrete_treatment=False)
    def fit(self, X, T, Y):
        self.model.fit(Y, T, X=X)

    def predict(self, X, T):
        base_Y = self.model.models_y[0][0].predict(X)
        base_A = self.model.models_t[0][0].predict(X)
        Y = self.model.effect(X, T0=base_A, T1=T) + base_Y

        return Y

class CFDML:
    def __init__(self):
        from econml.dml import CausalForestDML
        from sklearn.linear_model import LassoCV
        lasso_model = LassoCV(max_iter=5000, tol=1e-2, eps=1e-2, n_jobs=-1)
        self.model = CausalForestDML(model_y=lasso_model,
                                    model_t=lasso_model,
                                    discrete_treatment=False)
    def fit(self, X, T, Y):
        self.model.fit(Y, T, X=X)

    def predict(self, X, T):
        base_Y = self.model.models_y[0][0].predict(X)
        base_A = self.model.models_t[0][0].predict(X)
        Y = self.model.effect(X, T0=base_A, T1=T) + base_Y

        return Y

class LRL:
    def __init__(self):
        from sklearn.linear_model import LassoCV
        self.model = LassoCV(max_iter=5000, tol=1e-2)

    def fit(self, X, T, Y):
        self.model.fit(np.hstack([X, T.reshape(-1, 1)]), Y)

    def predict(self, X, T):
        if np.isscalar(T):
            T_col = np.full((X.shape[0], 1), T)
        else:
            T_col = np.array(T).reshape(-1, 1)

        return self.model.predict(np.hstack([X, T_col]))   
     
class Base:
    def __init__(self):
        pass

    def fit(self, X, T, Y):
        self.base = np.mean(Y)

    def predict(self, X, T):
        return self.base
    
def evaluate(model, data_dict, dataset_type, train_size=0.9):
    n = data_dict['x'].shape[0]

    indices = np.random.permutation(n)
    train_end = int(train_size * n)

    idx_train = indices[:train_end]
    idx_test = indices[train_end:]

    model.fit(data_dict['x'][idx_train], data_dict['a'][idx_train], data_dict['yf'][idx_train])

    grid_size = 2 ** 6 + 1
    dx = 1. / (grid_size - 1)
    treat_grid = np.linspace(np.finfo(float).eps, 1.0, grid_size)
    coefs = (data_dict["treat_coef"], data_dict["out_coef"])

    def compute_metrics(x_set):
        pred_list = [model.predict(x_set, treat) for treat in treat_grid]
        pred_grid = np.column_stack(pred_list)

        if dataset_type == 'ihdp':
            fact_list = [data.get_effect_ihdp(x_set, treat, coefs) for treat in treat_grid]
        else:
            fact_list = [data.get_effect_synt(x_set, treat, coefs) for treat in treat_grid]
        fact_grid = np.column_stack(fact_list)

        diff_sq = (fact_grid - pred_grid) ** 2
        mise = np.mean([romb(diff_sq[idx], dx=dx) for idx in range(x_set.shape[0])])
        adrfe = romb((fact_grid.mean(axis=0) - pred_grid.mean(axis=0)) ** 2, dx=dx)

        return np.sqrt(mise), np.sqrt(adrfe), fact_grid.mean(axis=0), pred_grid.mean(axis=0)

    mise_in, adrfe_in, f_in, p_in = compute_metrics(data_dict['x'][idx_train])
    mise_out, adrfe_out, f_out, p_out = compute_metrics(data_dict['x'][idx_test])

    return mise_in, adrfe_in, mise_out, adrfe_out, f_out, p_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    parser.add_argument('--model', type=str, default='VCNet', choices=['Base', 'LDML', 'CFDML', 'LRL', 'VCNet', 'DRNet', 'SCIGAN', 'DDMLCT', 'GIKS', 'ACFR', 'CSB'])
    args = parser.parse_args()
    dataset_type = args.dataset
    model_type = args.model

    if model_type == 'Base':
        model = Base()
    elif model_type == 'LDML':
        model = LDML()
    elif model_type == 'CFDML':
        model = CFDML()
    elif model_type == 'LRL':
        model = LRL()
    elif model_type == 'VCNet':
        from baseline_vcnet import VCNetWrapper
        # Dim = 25 for IHDP, 100 for Synt
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = VCNetWrapper(num_features=dim_x, model_name='Vcnet_tr')
    elif model_type == 'DRNet':
        from baseline_vcnet import VCNetWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = VCNetWrapper(num_features=dim_x, model_name='Drnet_tr')
    elif model_type == 'SCIGAN':
        from baseline_scigan import SCIGANWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = SCIGANWrapper(num_features=dim_x)
    elif model_type == 'DDMLCT':
        from baseline_ddmlct import DDMLCTWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = DDMLCTWrapper(num_features=dim_x)
    elif model_type == 'GIKS':
        from baseline_giks import GIKSWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = GIKSWrapper(num_features=dim_x)
    elif model_type == 'ACFR':
        from baseline_acfr import ACFRWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = ACFRWrapper(num_features=dim_x)
    elif model_type == 'CSB':
        from baseline_csb import CSBWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = CSBWrapper(num_features=dim_x)

    mise_in_list = []
    mise_out_list = []
    adrfe_in_list = []
    adrfe_out_list = []

    all_f_out = []
    all_p_out = []

    for i in range(10):
        data_name = f'./data/ihdp_semi_{i}.pkl' if dataset_type == 'ihdp' else f'./data/cont_synthetic_{i}.pkl'

        with open(data_name, 'rb') as file:
            try:
                loaded_data = pickle.load(file)
            except ModuleNotFoundError as e:
                if 'numpy._core' in str(e):
                    import sys
                    import numpy.core
                    sys.modules['numpy._core'] = numpy.core
                    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
                    file.seek(0)
                    loaded_data = pickle.load(file)
                else:
                    raise

        mise_in, adrfe_in, mise_out, adrfe_out, f_out, p_out = evaluate(model, loaded_data, dataset_type)
        mise_in_list.append(mise_in)
        adrfe_in_list.append(adrfe_in)
        mise_out_list.append(mise_out)
        adrfe_out_list.append(adrfe_out)

        all_f_out.append(f_out)
        all_p_out.append(p_out)

        print(f"[{model_type} - {dataset_type}] {i}th data\nIn-Sample mise: {mise_in:.4f}, adrfe: {adrfe_in:.4f}, \nOut-Sample mise: {mise_out:.4f}, adrfe: {adrfe_out:.4f}")

    avg_mise_in = np.mean(mise_in_list)
    avg_mise_out = np.mean(mise_out_list)
    avg_adrfe_in = np.mean(adrfe_in_list)
    avg_adrfe_out = np.mean(adrfe_out_list)

    print(f"[{model_type} - {dataset_type}] Average mise in: {avg_mise_in:.4f}, Average adrfe in: {avg_adrfe_in:.4f}")
    print(f"[{model_type} - {dataset_type}] Average mise out: {avg_mise_out:.4f}, Average adrfe out: {avg_adrfe_out:.4f}")

    grid_size = 2 ** 6 + 1
    treat_grid = np.linspace(np.finfo(float).eps, 1.0, grid_size)
    utils.plot_drf(treat_grid, np.mean(all_f_out, axis=0), np.mean(all_p_out, axis=0), f"{model_type.lower()}_{dataset_type.lower()}")