import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader
import pickle
import argparse
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import model
import utils

def objective(trial, dataset_type, data_splits, metric):
    trial.set_user_attr("mode", "multi" if len(data_splits) > 1 else "single")
    # Search Space
    lr_p = trial.suggest_categorical("lr_p", [1e-4, 5e-4, 1e-3, 5e-3])
    lr_s = trial.suggest_categorical("lr_s", [1e-3, 5e-3, 1e-2])
    epoch_total = trial.suggest_categorical("epoch_total", [300, 400, 500])
    dim_layer = trial.suggest_categorical("dim_layer", [32, 64, 128, 256])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2])
    
    weight_hsic = trial.suggest_categorical("weight_hsic", [0.5, 1.0, 2.5, 5.0, 10.0])
    coef_loss_c = trial.suggest_categorical("coef_loss_c", [0.01, 0.05, 0.1, 0.25, 0.5])
    coef_loss_p = trial.suggest_categorical("coef_loss_p", [0.01, 0.05, 0.1, 0.25, 0.5])
    weight_corr = trial.suggest_categorical("weight_corr", [1.0, 10.0, 50.0, 100.0, 500.0])
    init_logit = trial.suggest_categorical("init_logit", [10.0])

    hparams = {
        "weight_hsic": weight_hsic,
        "coef_loss_c": coef_loss_c,
        "coef_loss_p": coef_loss_p,
        "weight_corr": weight_corr,
        "lr_s": lr_s,
        "lr_p": lr_p,
        "dim_layer": dim_layer,
        "epoch_total": epoch_total,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "init_logit": init_logit,
    }

    metrics_list = []
    for set_train, set_val, set_test, coefs, info, t_range in data_splits:
        # Data Loading
        loader_train = DataLoader(set_train, batch_size=hparams["batch_size"], shuffle=True)
        loader_val = DataLoader(set_val, batch_size=hparams["batch_size"], shuffle=False)
        # We no longer load loader_test here to prevent test set leakage during tuning

        # Model Initialization
        selector = model.VSLayer(info, hparams)
        predictor_y = model.POLayer(info, hparams)
        model_main = model.MainModel(info={"vsl": selector, "pol": predictor_y}, hparams={"epoch_total": hparams["epoch_total"]})
        
        param_s = list(selector.parameters())
        param_p = list(predictor_y.parameters())

        optimizer_s = torch.optim.Adam(param_s, lr=hparams["lr_s"])
        optimizer_p = torch.optim.Adam(param_p, lr=hparams["lr_p"], weight_decay=hparams["weight_decay"])

        scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=100, gamma=0.97)
        scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=100, gamma=0.97)

        coef_loss_list = [hparams["weight_hsic"], hparams["coef_loss_c"], hparams["coef_loss_p"]]

        # We redirect stdout so that printing thousands of epochs doesn't flood the console
        sys.stdout = open(os.devnull, 'w')
        try:
            # Train and get the validation history
            _, _, history_val_y = model.train_model(
                model_main, optimizer_s, optimizer_p, scheduler_s, scheduler_p, 
                loader_train, loader_val, coef_loss_list
            )
            val_loss = history_val_y[-1]
            
            # Evaluate out-sample MISE on Validation set (Oracle Tuning)
            mise_val, _, _, _, _, _ = model.evaluate(model_main, loader_val, coefs, dataset_type, t_range=t_range)
        finally:
            sys.stdout = sys.__stdout__

        if metric == 'val_loss':
            metrics_list.append(val_loss)
        else:
            metrics_list.append(mise_val)

    # Optuna minimizes the chosen metric on the validation set
    return sum(metrics_list) / len(metrics_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    parser.add_argument('--metric', type=str, default='val_loss', choices=['val_loss', 'val_mise'], help='Metric to minimize during validation')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--multi', action='store_true', help='Evaluate on 3 datasets directly inside Optuna objective')
    args = parser.parse_args()
    dataset_type = args.dataset

    data_splits = []
    dataset_indices = [0, 1, 2] if args.multi else [0]
    
    for ds_idx in dataset_indices:
        data_path = os.path.join(parent_dir, 'data', f'ihdp_semi_{ds_idx}.pkl') if args.dataset == 'ihdp' else os.path.join(parent_dir, 'data', f'cont_synthetic_{ds_idx}.pkl')
        with open(data_path, 'rb') as file:
            try:
                data = pickle.load(file)
            except Exception as e:
                print("ERROR", e)

        # Split data ONCE before tuning starts to ensure fair comparison
        dataset = {k: data[k] for k in data.keys()}
        set_train, set_val, set_test, coefs = utils.split_data(dataset, train_size=0.63, val_size=0.27)
        
        info = {
            "dim_x": data['x'].shape[1],
            "dim_a": 1,
            "dim_y": 1,
            "HSIC_xa": coefs[2]
        }
        t_range = (np.percentile(data['a'], 5), np.percentile(data['a'], 95))
        data_splits.append((set_train, set_val, set_test, coefs, info, t_range))

    print(f"Starting Optuna hyperparameter tuning for {args.dataset} dataset ({args.trials} trials)...")
    if args.multi:
        print(f"multi is ON: Optimizing over average {args.metric} of 3 datasets.")
    
    # Start Tuning (Using SQLite storage to prevent data loss if interrupted)
    study_name = f"cvscen_{args.dataset}_{args.metric}_study"
    db_path = os.path.join(current_dir, f"optuna_study_CVSCEN_{args.dataset}_{args.metric}.db")
    storage_name = f"sqlite:///{db_path}"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True)
    
    # Enqueue existing best parameters so Optuna evaluates them first
    existing_hparams_path = os.path.join(current_dir, f'best_hparams_CVSCEN_{args.dataset}_{args.metric}.json')
    if os.path.exists(existing_hparams_path):
        try:
            with open(existing_hparams_path, 'r') as f:
                old_best = json.load(f)
            study.enqueue_trial(old_best, skip_if_exists=False)
            print(f"Enqueued existing best parameters from {existing_hparams_path} to evaluate as a baseline.")
        except Exception as e:
            print(f"Failed to enqueue existing hparams: {e}")

    try:
        study.optimize(lambda trial: objective(trial, dataset_type, data_splits, args.metric), n_trials=args.trials)
    except KeyboardInterrupt:
        print("\n[!] Tuning interrupted by user! Saving best parameters found so far...")

    if len(study.trials) > 0:
        if args.multi:
            multi_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("mode") == "multi"]
            if len(multi_trials) == 0:
                print("\nNo multi-mode trials completed. Cannot save robust best parameters.")
            else:
                multi_trials.sort(key=lambda t: t.value)
                best_multi_trial = multi_trials[0]
                true_best_params = best_multi_trial.params
                best_avg_metric = best_multi_trial.value
                print("\n==============================================")
                print(f"True Best Trial Selected directly from Multi-mode Trials (Evaluated on {len(data_splits)} datasets)")
                print(f"True Best Avg Validation {args.metric}: {best_avg_metric:.4f}")
                print("Params:")
                for key, value in true_best_params.items():
                    print(f"    {key}: {value}")
                    
                out_file = os.path.join(current_dir, f'best_hparams_CVSCEN_{args.dataset}_{args.metric}.json')
                with open(out_file, 'w') as f:
                    json.dump(true_best_params, f, indent=4)
                print(f"\nRobust best parameters successfully saved to {out_file}.")
        else:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            completed_trials.sort(key=lambda t: t.value)
            
            # 중복 파라미터 조합 제거 (Optuna가 똑같은 파라미터를 여러 번 탐색했을 경우 방지)
            unique_trials = []
            seen_params = set()
            for t in completed_trials:
                param_tuple = tuple(sorted(t.params.items()))
                if param_tuple not in seen_params:
                    seen_params.add(param_tuple)
                    unique_trials.append(t)
            
            top_k = min(3, len(unique_trials)) # Top 3 재평가
            print(f"\n--- Top {top_k} Re-evaluation (Robustness Check for Unique Params) ---")
            
            best_avg_metric = float('inf')
            true_best_params = None
            true_best_trial_number = -1
            
            for i, trial in enumerate(unique_trials[:top_k]):
                print(f"\nRe-evaluating Trial {trial.number} (Optuna Valid {args.metric}: {trial.value:.4f})")
                print(f"  Params: {trial.params}")
                
                # Cross-Dataset Robustness Check (Evaluate on Datasets 1, 2, 3)
                data_splits_reval = []
                for ds_idx in [1, 2, 3]:
                    import random
                    random.seed(42)
                    torch.manual_seed(42)
                    np.random.seed(42)
                    
                    ds_path = os.path.join(parent_dir, 'data', f'ihdp_semi_{ds_idx}.pkl') if args.dataset == 'ihdp' else os.path.join(parent_dir, 'data', f'cont_synthetic_{ds_idx}.pkl')
                    with open(ds_path, 'rb') as f:
                        data_k = pickle.load(f)
                    dataset_k = {k: data_k[k] for k in data_k.keys()}
                    set_train_k, set_val_k, set_test_k, coefs_k = utils.split_data(dataset_k, train_size=0.63, val_size=0.27)
                    info_k = {
                        "dim_x": data_k['x'].shape[1],
                        "dim_a": 1,
                        "dim_y": 1,
                        "HSIC_xa": coefs_k[2]
                    }
                    t_range_k = (np.percentile(data_k['a'], 5), np.percentile(data_k['a'], 95))
                    data_splits_reval.append((set_train_k, set_val_k, set_test_k, coefs_k, info_k, t_range_k))
                
                fixed_trial = optuna.trial.FixedTrial(trial.params)
                avg_metric = objective(fixed_trial, dataset_type, data_splits_reval, args.metric)
                print(f"  -> Avg Validation {args.metric} over {len(data_splits_reval)} Datasets: {avg_metric:.4f}")
                
                if avg_metric < best_avg_metric:
                    best_avg_metric = avg_metric
                    true_best_params = trial.params
                    true_best_trial_number = trial.number
                    
            print("\n==============================================")
            print(f"True Best Trial Selected: {true_best_trial_number}")
            print(f"True Best Avg Validation {args.metric}: {best_avg_metric:.4f}")
            print("Params:")
            for key, value in true_best_params.items():
                print(f"    {key}: {value}")

            # Save to JSON
            out_file = os.path.join(current_dir, f'best_hparams_CVSCEN_{args.dataset}_{args.metric}.json')
            with open(out_file, 'w') as f:
                json.dump(true_best_params, f, indent=4)
            print(f"\nRobust best parameters successfully saved to {out_file}.")
    else:
        print("\nNo trials completed. Nothing to save.")
