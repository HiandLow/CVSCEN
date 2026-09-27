import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import optuna
import numpy as np
import pickle
import argparse
import json
from scipy.integrate import romb
import data

def evaluate_val_loss(model, X_val, T_val, Y_val):
    # Depending on the model, calculate factual MSE
    preds = model.predict(X_val, T_val)
    val_loss = np.mean((preds - Y_val) ** 2)
    return val_loss

def evaluate_val_mise(model, data_dict, dataset_type, X_val, adrf_only=False):
    # Calculate Oracle MISE for validation
    grid_size = 2 ** 6 + 1
    t_min, t_max = np.percentile(data_dict['a'], 5), np.percentile(data_dict['a'], 95)
    dx = (t_max - t_min) / (grid_size - 1)
    treat_grid = np.linspace(t_min, t_max, grid_size)
    coefs = (data_dict["treat_coef"], data_dict["out_coef"])

    pred_list = [model.predict(X_val, treat) for treat in treat_grid]
    pred_grid = np.column_stack(pred_list)

    if dataset_type == 'ihdp':
        fact_list = [data.get_effect_ihdp(X_val, treat, coefs) for treat in treat_grid]
    else:
        fact_list = [data.get_effect_synt(X_val, treat, coefs) for treat in treat_grid]
    
    fact_grid = np.column_stack(fact_list)
    if adrf_only:
        # Population-level estimators (DDMLCT) are scored on the ADRF, not per-unit curves
        return np.sqrt(romb((fact_grid.mean(axis=0) - pred_grid.mean(axis=0)) ** 2, dx=dx))
    diff_sq = (fact_grid - pred_grid) ** 2
    mise = np.mean([romb(diff_sq[idx], dx=dx) for idx in range(X_val.shape[0])])
    return np.sqrt(mise)

def objective(trial, model_type, dataset_type, data_splits, metric='val_loss'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    dim_x = 25 if dataset_type == 'ihdp' else 100
    
    trial.set_user_attr("mode", "multi" if len(data_splits) > 1 else "single")
    
    model_type_lower = model_type.lower()
    kwargs = {}
    
    # Conditional Common search space for non-DDMLCT
    if model_type_lower not in ['ddmlct', 'scigan']:
        # DRNet's original learning rate (0.05) is included in its own grid
        lr_choices = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2] + ([5e-2] if model_type_lower == 'drnet' else [])
        kwargs['lr'] = trial.suggest_categorical("lr" if model_type_lower != 'drnet' else "lr_drnet", lr_choices)
        kwargs['weight_decay'] = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 5e-3, 1e-2])

    # 1. Base ranges for common parameters
    batch_choices = [32, 64, 128]
    dim_choices = [32, 64, 128, 256]
    epoch_choices = [300, 400, 500]
    
    # Names for backward compatibility with existing Optuna DBs
    epoch_name = "epoch_total"
    dim_name = "dim_layer"
    batch_name = "batch_size"

    # 2. Override ranges for specific models
    if model_type_lower == 'scigan':
        batch_choices = [16, 32, 64, 128, 256]
        batch_name = "batch_size_scigan"
    elif model_type_lower == 'csb':
        dim_choices = [128, 256, 512]
        epoch_choices = [1000, 2000, 3000]
        dim_name = "dim_layer_csb"
        epoch_name = "epoch_total_csb"
    elif model_type_lower in ['vcnet', 'drnet']:
        epoch_choices = [300, 500, 700, 800]
        epoch_name = f"epoch_total_{model_type_lower}"
    elif model_type_lower == 'giks':
        epoch_choices = [200, 300, 400, 500]
    elif model_type_lower == 'ddmlct':
        dim_choices = [10, 25, 64, 100, 128]
        dim_name = "dim_layer_ddmlct"
        
    # 3. Suggest common parameters exactly once
    if model_type_lower != 'ddmlct':
        kwargs['batch_size'] = trial.suggest_categorical(batch_name, batch_choices)
    kwargs['epoch_total'] = trial.suggest_categorical(epoch_name, epoch_choices)
    kwargs['dim_layer'] = trial.suggest_categorical(dim_name, dim_choices)
    
    # Pre-suggest DDMLCT unique parameters if needed so they are registered for the trial
    if model_type_lower == 'ddmlct':
        kwargs['lr_s'] = trial.suggest_categorical("lr_s", [0.001, 0.01, 0.05, 0.15, 0.4])
        kwargs['L'] = trial.suggest_categorical("L", [2, 5, 10])
        kwargs['lr'] = trial.suggest_categorical("lr_ddmlct", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 0.15, 0.4])
        kwargs['weight_decay'] = trial.suggest_categorical("weight_decay_ddmlct", [1e-4, 1e-3, 5e-3, 1e-2, 0.1, 0.2, 0.3])
    elif model_type_lower == 'acfr':
        kwargs['lr_s'] = trial.suggest_categorical("lr_s", [1e-5, 1e-4, 1e-3])
        kwargs['gamma1'] = trial.suggest_categorical("gamma1", [0.1, 1.0, 5.0, 10.0])
        kwargs['gamma2'] = trial.suggest_categorical("gamma2", [0.1, 0.2, 0.5])
        kwargs['std'] = trial.suggest_categorical("std", [0.1, 0.2, 0.5])
        kwargs['m'] = trial.suggest_categorical("m", [1, 5, 10, 20])
    elif model_type_lower == 'scigan':
        kwargs['alpha'] = trial.suggest_categorical("alpha", [0.1, 1.0, 10.0])
        kwargs['num_dosage_samples'] = trial.suggest_categorical("num_dosage_samples", [3, 5, 10])
        kwargs['lr_g'] = trial.suggest_categorical("lr_g", [1e-4, 5e-4, 1e-3])
        kwargs['lr_d'] = trial.suggest_categorical("lr_d", [1e-4, 5e-4, 1e-3])
        kwargs['d_steps'] = trial.suggest_categorical("d_steps", [1, 3, 5])
    elif model_type_lower == 'giks':
        kwargs['gp_lambda'] = trial.suggest_categorical("gp_lambda", [1e-2, 1e-1, 1.0])
        kwargs['gi_linear_delta'] = trial.suggest_categorical("gi_linear_delta", [0.01, 0.05, 0.1])
        kwargs['gp_linear_delta'] = trial.suggest_categorical("gp_linear_delta", [0.05, 0.1, 0.2])
        kwargs['num_grid'] = trial.suggest_categorical("num_grid", [5, 10, 20])
        kwargs['degree'] = trial.suggest_categorical("degree", [2, 3])
    elif model_type_lower == 'csb':
        kwargs['beta'] = trial.suggest_categorical("beta_csb", [0.001, 0.01, 0.1, 1.0])
        kwargs['gamma'] = trial.suggest_categorical("gamma_csb", [0.001, 0.01, 0.1, 1.0])
        kwargs['kappa'] = trial.suggest_categorical("kappa", [0.0, 0.5, 1.0])
        kwargs['z_dim'] = trial.suggest_categorical("z_dim", [16, 32, 64])
        kwargs['t_dim_latent'] = trial.suggest_categorical("t_dim_latent", [4, 8, 16])
    elif model_type_lower in ['vcnet', 'drnet']:
        kwargs['alpha'] = trial.suggest_categorical("alpha", [0.1, 0.5, 1.0, 2.0])
        kwargs['num_grid'] = trial.suggest_categorical("num_grid", [5, 10, 20])
        kwargs['degree'] = trial.suggest_categorical("degree", [2, 3])
        kwargs['tr_lr'] = trial.suggest_categorical("tr_lr", [1e-4, 1e-3, 1e-2])
    
    val_scores = []
    
    sys.stdout = open(os.devnull, 'w')
    try:
        for X_train, T_train, Y_train, X_val, T_val, Y_val, loaded_data in data_splits:
            # Re-initialize wrapper per dataset to reset weights
            if model_type_lower in ['vcnet', 'drnet']:
                from baselines.vcnet.baseline_vcnet import VCNetWrapper
                model_name = 'Vcnet_tr' if model_type_lower == 'vcnet' else 'Drnet_tr'
                model = VCNetWrapper(num_features=dim_x, model_name=model_name, **kwargs)
            elif model_type_lower == 'acfr':
                from baselines.acfr.baseline_acfr import ACFRWrapper
                model = ACFRWrapper(num_features=dim_x, **kwargs)
            elif model_type_lower == 'scigan':
                from baselines.scigan.baseline_scigan import SCIGANWrapper
                model = SCIGANWrapper(num_features=dim_x, **kwargs)
            elif model_type_lower == 'giks':
                from baselines.giks.baseline_giks import GIKSWrapper
                model = GIKSWrapper(num_features=dim_x, **kwargs)
            elif model_type_lower == 'csb':
                from baselines.csb.baseline_csb import CSBWrapper
                model = CSBWrapper(num_features=dim_x, **kwargs)
            elif model_type_lower == 'ddmlct':
                from baselines.ddmlct.baseline_ddmlct import DDMLCTWrapper
                model = DDMLCTWrapper(num_features=dim_x, **kwargs)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            model.fit(X_train, T_train, Y_train)
            if metric == 'val_mise':
                score = evaluate_val_mise(model, loaded_data, dataset_type, X_val, adrf_only=(model_type_lower == 'ddmlct'))
            else:
                score = evaluate_val_loss(model, X_val, T_val, Y_val)
            val_scores.append(score)
            
        avg_val_score = np.mean(val_scores)
    except Exception as e:
        import traceback
        print(f"[{model_type}] trial failed, scoring 1e6:\n{traceback.format_exc()}", file=sys.stderr)
        avg_val_score = 1e6 # Penalize failed runs
    finally:
        sys.stdout = sys.__stdout__
        
    return avg_val_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--metric', type=str, default='val_loss', choices=['val_loss', 'val_mise'])
    parser.add_argument('--multi', action='store_true', help='Evaluate on 3 datasets directly inside Optuna objective')
    args = parser.parse_args()

    models_to_tune = ['VCNet', 'DRNet', 'ACFR', 'SCIGAN', 'GIKS', 'CSB', 'DDMLCT']
    if args.model != 'all':
        models_to_tune = [args.model]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    dataset_indices = [0, 1, 2] if args.multi else [0]
    data_splits = []
    
    for ds_idx in dataset_indices:
        data_name = os.path.join(parent_dir, 'data', f'ihdp_semi_{ds_idx}.pkl') if args.dataset == 'ihdp' else os.path.join(parent_dir, 'data', f'cont_synthetic_{ds_idx}.pkl')
        with open(data_name, 'rb') as file:
            try:
                loaded_data = pickle.load(file)
            except ModuleNotFoundError:
                import numpy.core
                sys.modules['numpy._core'] = numpy.core
                sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
                file.seek(0)
                loaded_data = pickle.load(file)

        n = loaded_data['x'].shape[0]
        indices = np.random.RandomState(42).permutation(n)
        
        train_end = int(0.63 * n)
        val_end = train_end + int(0.27 * n)
        
        idx_train = indices[:train_end]
        idx_val = indices[train_end:val_end]
        
        X_train = loaded_data['x'][idx_train]
        T_train = loaded_data['a'][idx_train]
        Y_train = loaded_data['yf'][idx_train]

        X_val = loaded_data['x'][idx_val]
        T_val = loaded_data['a'][idx_val]
        Y_val = loaded_data['yf'][idx_val]

        data_splits.append((X_train, T_train, Y_train, X_val, T_val, Y_val, loaded_data))

    for model_type in models_to_tune:
        print(f"\nStarting Optuna tuning for {model_type} on {args.dataset} ({args.trials} trials) minimizing {args.metric}...")
        if args.multi:
            print(f"multi is ON: Optimizing over average {args.metric} of 3 datasets.")
        
        os.makedirs("tune", exist_ok=True)
        db_path = os.path.join(current_dir, f"optuna_{model_type}_{args.dataset}_{args.metric}.db")
        storage_url = f"sqlite:///{db_path}"
        study_name = f"{model_type}_{args.dataset}_{args.metric}_tuning"
        
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_url, 
            direction="minimize", 
            load_if_exists=True
        )

        existing_hparams_path = os.path.join(current_dir, f'best_hparams_{model_type}_{args.dataset}_{args.metric}.json')
        if os.path.exists(existing_hparams_path):
            try:
                with open(existing_hparams_path, 'r') as f:
                    old_best = json.load(f)
                
                if model_type.lower() == 'drnet' and 'epoch_total_vcnet' in old_best:
                    old_best['epoch_total_drnet'] = old_best.pop('epoch_total_vcnet')
                
                study.enqueue_trial(old_best, skip_if_exists=False)
                print(f"Enqueued existing best parameters from {existing_hparams_path} to evaluate as a baseline.")
            except Exception as e:
                print(f"Failed to enqueue existing hparams: {e}")

        try:
            study.optimize(
                lambda trial: objective(trial, model_type, args.dataset, data_splits, args.metric),
                n_trials=args.trials,
                catch=(Exception,)
            )
        except KeyboardInterrupt:
            print("\nOptimization interrupted. Saving best parameters so far...")

        try:
            if args.multi:
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("mode") == "multi"]
                if len(completed_trials) == 0:
                    print(f"\nNo multi-mode trials completed for {model_type}. Cannot save robust best parameters.")
                    continue
            else:
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if len(completed_trials) == 0:
                    print(f"\nNo trials completed for {model_type}. Cannot save parameters.")
                    continue
            
            completed_trials.sort(key=lambda t: t.value)
            
            unique_trials = []
            seen_params = set()
            for t in completed_trials:
                param_tuple = tuple(sorted(t.params.items()))
                if param_tuple not in seen_params:
                    seen_params.add(param_tuple)
                    unique_trials.append(t)
            
            top_k = min(3, len(unique_trials))
            
            # If multi is on, the top trial is already robustly evaluated. Just save it directly.
            if args.multi:
                print(f"\n--- True Best Trial for {model_type} (from multi-mode) ---")
                best_trial = unique_trials[0]
                best_avg_metric = best_trial.value
                true_best_params = best_trial.params
                true_best_trial_number = best_trial.number
            else:
                print(f"\n--- Top {top_k} Re-evaluation (Robustness Check for {model_type}) ---")
                best_avg_metric = float('inf')
                true_best_params = None
                true_best_trial_number = -1
                
                for i, trial in enumerate(unique_trials[:top_k]):
                    print(f"\nRe-evaluating Trial {trial.number} (Valid Score on split 0: {trial.value:.4f})")
                    
                    # Cross-Dataset Robustness Check on splits 1, 2, 3
                    data_splits_reval = []
                    for ds_idx in [1, 2, 3]:
                        import random
                        random.seed(42)
                        import torch
                        torch.manual_seed(42)
                        np.random.seed(42)

                        ds_path = os.path.join(parent_dir, 'data', f'ihdp_semi_{ds_idx}.pkl') if args.dataset == 'ihdp' else os.path.join(parent_dir, 'data', f'cont_synthetic_{ds_idx}.pkl')
                        with open(ds_path, 'rb') as f:
                            data_k = pickle.load(f)
                        
                        n_k = data_k['x'].shape[0]
                        indices_k = np.random.RandomState(42).permutation(n_k)
                        train_end_k = int(0.63 * n_k)
                        val_end_k = train_end_k + int(0.27 * n_k)
                        
                        idx_train_k = indices_k[:train_end_k]
                        idx_val_k = indices_k[train_end_k:val_end_k]
                        
                        X_train_k = data_k['x'][idx_train_k]
                        T_train_k = data_k['a'][idx_train_k]
                        Y_train_k = data_k['yf'][idx_train_k]

                        X_val_k = data_k['x'][idx_val_k]
                        T_val_k = data_k['a'][idx_val_k]
                        Y_val_k = data_k['yf'][idx_val_k]
                        
                        data_splits_reval.append((X_train_k, T_train_k, Y_train_k, X_val_k, T_val_k, Y_val_k, data_k))
                        
                    fixed_trial = optuna.trial.FixedTrial(trial.params)
                    avg_metric = objective(fixed_trial, model_type, args.dataset, data_splits_reval, args.metric)
                    print(f"  -> Avg Validation Score over {len(data_splits_reval)} Datasets: {avg_metric:.4f}")
                    
                    if avg_metric < best_avg_metric:
                        best_avg_metric = avg_metric
                        true_best_params = trial.params
                        true_best_trial_number = trial.number
            
            print(f"\n==============================================")
            print(f"True Best Trial Selected: {true_best_trial_number}")
            print(f"True Best Avg Validation {args.metric}: {best_avg_metric:.4f}")
            
            true_best_params = dict(true_best_params)
            if model_type.lower() == 'drnet' and 'epoch_total_vcnet' in true_best_params:
                true_best_params['epoch_total_drnet'] = true_best_params.pop('epoch_total_vcnet')

            print("Params:")
            for key, value in true_best_params.items():
                print(f"    {key}: {value}")

            hparams_path = os.path.join(current_dir, f'best_hparams_{model_type}_{args.dataset}_{args.metric}.json')
            with open(hparams_path, 'w') as f:
                json.dump(true_best_params, f, indent=4)
            print(f"\nRobust best parameters successfully saved to {hparams_path}.")
        except Exception as e:
            print(f"\nCould not retrieve robust best trial for {model_type}: {e}")

if __name__ == "__main__":
    main()
