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
    
def evaluate(model, data_dict, dataset_type, train_size=0.63, val_size=0.27):
    n = data_dict['x'].shape[0]

    indices = np.random.RandomState(42).permutation(n)
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)

    idx_train = indices[:train_end]
    idx_test = indices[val_end:]

    model.fit(data_dict['x'][idx_train], data_dict['a'][idx_train], data_dict['yf'][idx_train])

    grid_size = 2 ** 6 + 1
    t_min, t_max = np.percentile(data_dict['a'], 5), np.percentile(data_dict['a'], 95)
    dx = (t_max - t_min) / (grid_size - 1)
    treat_grid = np.linspace(t_min, t_max, grid_size)
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

    import json
    import os
    tune_params = {}
    best_hparams_file = os.path.join('tune', f"best_hparams_{model_type}_{dataset_type}.json")
    if os.path.exists(best_hparams_file):
        print(f"Loading tuned hyperparameters from {best_hparams_file}...")
        with open(best_hparams_file, "r") as f:
            raw_params = json.load(f)
            
            # Remap model-specific parameters to override generic ones
            suffix = f"_{model_type.lower()}"
            for k, v in raw_params.items():
                if k.endswith(suffix):
                    tune_params[k.replace(suffix, "")] = v
            for k, v in raw_params.items():
                if not k.endswith(suffix) and k not in tune_params:
                    tune_params[k] = v
    else:
        print(f"No tuned hyperparameters found at {best_hparams_file}, using defaults.")

    if model_type == 'Base':
        model = Base()
    elif model_type == 'LDML':
        model = LDML()
    elif model_type == 'CFDML':
        model = CFDML()
    elif model_type == 'LRL':
        model = LRL()
    elif model_type == 'VCNet':
        from baselines.vcnet.baseline_vcnet import VCNetWrapper
        # Dim = 25 for IHDP, 100 for Synt
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = VCNetWrapper(num_features=dim_x, model_name='Vcnet_tr', **tune_params)
    elif model_type == 'DRNet':
        from baselines.vcnet.baseline_vcnet import VCNetWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = VCNetWrapper(num_features=dim_x, model_name='Drnet_tr', **tune_params)
    elif model_type == 'SCIGAN':
        from baselines.scigan.baseline_scigan import SCIGANWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = SCIGANWrapper(num_features=dim_x, **tune_params)
    elif model_type == 'DDMLCT':
        from baselines.ddmlct.baseline_ddmlct import DDMLCTWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = DDMLCTWrapper(num_features=dim_x, **tune_params)
    elif model_type == 'GIKS':
        from baselines.giks.baseline_giks import GIKSWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = GIKSWrapper(num_features=dim_x, **tune_params)
    elif model_type == 'ACFR':
        from baselines.acfr.baseline_acfr import ACFRWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = ACFRWrapper(num_features=dim_x, **tune_params)
    elif model_type == 'CSB':
        from baselines.csb.baseline_csb import CSBWrapper
        dim_x = 25 if dataset_type == 'ihdp' else 100
        model = CSBWrapper(num_features=dim_x, **tune_params)

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

    avg_mise_in, std_mise_in = np.mean(mise_in_list), np.std(mise_in_list)
    avg_mise_out, std_mise_out = np.mean(mise_out_list), np.std(mise_out_list)
    avg_adrfe_in, std_adrfe_in = np.mean(adrfe_in_list), np.std(adrfe_in_list)
    avg_adrfe_out, std_adrfe_out = np.mean(adrfe_out_list), np.std(adrfe_out_list)

    print(f"[{model_type} - {dataset_type}] Average In-Sample over all iterations\nmise: {avg_mise_in:.4f} ± {std_mise_in:.4f}, adrfe: {avg_adrfe_in:.4f} ± {std_adrfe_in:.4f}")
    print(f"[{model_type} - {dataset_type}] Average Out-Sample over all iterations\nmise: {avg_mise_out:.4f} ± {std_mise_out:.4f}, adrfe: {avg_adrfe_out:.4f} ± {std_adrfe_out:.4f}")

    grid_size = 2 ** 6 + 1
    t_min, t_max = np.percentile(loaded_data['a'], 5), np.percentile(loaded_data['a'], 95)
    treat_grid = np.linspace(t_min, t_max, grid_size)
    utils.plot_drf(treat_grid, np.mean(all_f_out, axis=0), np.mean(all_p_out, axis=0), f"{model_type.lower()}_{dataset_type.lower()}")