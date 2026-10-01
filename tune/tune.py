import contextlib
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

def objective(trial, dataset_type, data_splits, metric, guidance='logit', selector='role', seeds=None):
    trial.set_user_attr("mode", "multi" if len(data_splits) > 1 else "single")
    hparams = {
        "selector": selector,
        "lr_p": trial.suggest_categorical("lr_p", [1e-4, 5e-4, 1e-3, 5e-3]),
        "lr_s": trial.suggest_categorical("lr_s", [1e-3]),
        "epoch_total": trial.suggest_categorical("epoch_total", [300, 400, 500]),
        "dim_layer": trial.suggest_categorical("dim_layer", [32, 64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "weight_decay": trial.suggest_categorical("weight_decay", [1e-4, 1e-3, 1e-2]),
    }
    if selector == 'two_gate':
        hparams.update({
            "weight_treat": 1.0,
            "coef_loss_a": 0.1,
            "coef_loss_y": trial.suggest_categorical("coef_loss_y", [0.01, 0.05, 0.1, 0.25, 0.5]),
            "init_logit": trial.suggest_categorical("init_logit", [0.0, 1.0, 2.0]),
        })
    else:
        if guidance == 'search':
            guidance = trial.suggest_categorical("guidance", ["penalty", "logit"])
    if selector != 'two_gate' and guidance == 'calibrated':
        hparams.update({
            "guidance": guidance,
            "coef_loss": trial.suggest_categorical("coef_loss", [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04]),
            "init_logit": trial.suggest_categorical("init_logit", [0.0, 1.0, 2.0]),
            "weight_dep": 10.0,
        })
    elif selector != 'two_gate':
        hparams.update({
            "guidance": guidance,
            "weight_hsic": trial.suggest_categorical("weight_hsic", [0.5, 1.0, 2.5, 5.0]),
            "coef_loss_c": trial.suggest_categorical("coef_loss", [0.05, 0.1, 0.25, 0.5]),
            "weight_corr": trial.suggest_categorical("weight_corr", [0.5, 1.0, 2.5, 5.0]),
            "init_logit": trial.suggest_categorical("init_logit", [0.0, 0.5, 1.0]),
        })
        hparams["coef_loss_p"] = hparams["coef_loss_c"]

    metrics_list = []
    for k, (set_train, set_val, set_test, coefs, info, t_range, x_ref) in enumerate(data_splits):
        utils.set_seed(k if seeds is None else seeds[k])
        loader_train = DataLoader(set_train, batch_size=hparams["batch_size"], shuffle=True)
        loader_val = DataLoader(set_val, batch_size=hparams["batch_size"], shuffle=False)
        model_main, optimizer_s, optimizer_p, scheduler_s, scheduler_p, coef_loss_list = model.build(info, hparams)

        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
            _, _, history_val_y = model.train_model(
                model_main, optimizer_s, optimizer_p, scheduler_s, scheduler_p,
                loader_train, loader_val, coef_loss_list
            )
            if metric == 'val_loss':
                metrics_list.append(history_val_y[-1])
            else:
                mise_val = model.evaluate(model_main, loader_val, coefs, dataset_type, t_range=t_range, x_ref=x_ref)[0]
                metrics_list.append(mise_val)

    return sum(metrics_list) / len(metrics_list)

def one_se_path(params, distribution, dataset_type, data_splits, metric, guidance, selector):
    lams = sorted(distribution.choices)
    path = []
    for lam in lams:
        trial_params = dict(params, coef_loss=lam)
        values = [objective(optuna.trial.FixedTrial(trial_params), dataset_type, [split], metric, guidance, selector, seeds=[k])
                  for k, split in enumerate(data_splits)]
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        path.append({"coef_loss": lam, "mean": mean, "se": se, "values": [float(v) for v in values]})
        print(f"  coef_loss {lam:<8} mean {metric} {mean:.4f} (SE {se:.4f})")
    best = min(path, key=lambda p: p["mean"])
    threshold = best["mean"] + best["se"]
    chosen = max((p for p in path if p["mean"] <= threshold), key=lambda p: p["coef_loss"])
    return chosen["coef_loss"], best["coef_loss"], threshold, path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    parser.add_argument('--metric', type=str, default='val_loss', choices=['val_loss', 'val_mise'], help='Metric to minimize during validation')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--multi', action='store_true', help='Evaluate on 3 datasets directly inside Optuna objective')
    parser.add_argument('--guidance', type=str, default='logit', choices=['penalty', 'logit', 'search', 'calibrated'],
                        help="Dependence guidance: sparsity penalty, logit shift, or tuned as a hyperparameter")
    parser.add_argument('--selector', type=str, default='role', choices=['role', 'two_gate'],
                        help="role: 3-way role selector with HSIC guidance; two_gate: treatment and outcome gates")
    parser.add_argument('--skip-1se', action='store_true',
                        help="With --multi, keep the minimum-validation coef_loss instead of applying the one-standard-error rule")
    args = parser.parse_args()
    dataset_type = args.dataset
    if args.selector == 'two_gate':
        tag = "_two_gate"
    else:
        tag = "" if args.guidance == 'penalty' else f"_{args.guidance}"

    def save_best(params):
        params = dict(params)
        params["selector"] = args.selector
        if args.selector == 'role' and args.guidance != 'search':
            params["guidance"] = args.guidance
        out_file = os.path.join(current_dir, f'best_hparams_CVSCEN_{args.dataset}_{args.metric}{tag}.json')
        with open(out_file, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"\nRobust best parameters successfully saved to {out_file}.")

    data_splits = []
    dataset_indices = [0, 1, 2] if args.multi else [0]
    
    for ds_idx in dataset_indices:
        data_path = os.path.join(parent_dir, 'data', f'ihdp_semi_{ds_idx}.pkl') if args.dataset == 'ihdp' else os.path.join(parent_dir, 'data', f'cont_synthetic_{ds_idx}.pkl')
        with open(data_path, 'rb') as file:
            data = pickle.load(file)

        dataset = {k: data[k] for k in data.keys()}
        set_train, set_val, set_test, coefs = utils.split_data(dataset, train_size=0.63, val_size=0.27)
        
        info = {
            "dim_x": data['x'].shape[1],
            "dim_a": 1,
            "dim_y": 1,
            "HSIC_xa": utils.train_hsic(set_train),
            "dep_z": utils.cached_dependence_z(set_train, f"{args.dataset}_{ds_idx}"),
        }
        t_range = (np.percentile(data['a'], 5), np.percentile(data['a'], 95))
        data_splits.append((set_train, set_val, set_test, coefs, info, t_range, data['x']))

    print(f"Starting Optuna hyperparameter tuning for {args.dataset} dataset ({args.trials} trials)...")
    if args.multi:
        print(f"multi is ON: Optimizing over average {args.metric} of 3 datasets.")
    
    study_name = f"cvscen_{args.dataset}_{args.metric}{tag}_study"
    db_path = os.path.join(current_dir, f"optuna_study_CVSCEN_{args.dataset}_{args.metric}{tag}.db")
    storage_name = f"sqlite:///{db_path}"
    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))

    existing_hparams_path = os.path.join(current_dir, f'best_hparams_CVSCEN_{args.dataset}_{args.metric}{tag}.json')
    if os.path.exists(existing_hparams_path):
        try:
            with open(existing_hparams_path, 'r') as f:
                old_best = json.load(f)
            old_best.pop("selector", None)
            if args.guidance != 'search':
                old_best.pop("guidance", None)
            study.enqueue_trial(old_best, skip_if_exists=False)
            print(f"Enqueued existing best parameters from {existing_hparams_path} to evaluate as a baseline.")
        except Exception as e:
            print(f"Failed to enqueue existing hparams: {e}")

    try:
        study.optimize(lambda trial: objective(trial, dataset_type, data_splits, args.metric, args.guidance, args.selector), n_trials=args.trials)
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

                true_best_params = dict(true_best_params)
                if "coef_loss" in best_multi_trial.distributions and not args.skip_1se:
                    print("\n--- One-standard-error rule for coef_loss (same datasets and seeds as the multi trials) ---")
                    chosen, min_lam, threshold, path = one_se_path(
                        true_best_params, best_multi_trial.distributions["coef_loss"], dataset_type, data_splits,
                        args.metric, args.guidance, args.selector)
                    print(f"Minimum at coef_loss {min_lam}; threshold {threshold:.4f}; chosen coef_loss {chosen} "
                          f"(multi-phase pick was {true_best_params['coef_loss']})")
                    record = {"multi_pick": dict(true_best_params), "multi_pick_value": best_avg_metric,
                              "min_coef_loss": min_lam, "threshold": threshold, "chosen_coef_loss": chosen, "path": path}
                    with open(os.path.join(current_dir, f'one_se_path_CVSCEN_{args.dataset}_{args.metric}{tag}.json'), 'w') as f:
                        json.dump(record, f, indent=4)
                    true_best_params["coef_loss"] = chosen

                save_best(true_best_params)
        else:
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            completed_trials.sort(key=lambda t: t.value)
            
            unique_trials = []
            seen_params = set()
            for t in completed_trials:
                param_tuple = tuple(sorted(t.params.items()))
                if param_tuple not in seen_params:
                    seen_params.add(param_tuple)
                    unique_trials.append(t)
            
            top_k = min(3, len(unique_trials))
            print(f"\n--- Top {top_k} Re-evaluation (Robustness Check for Unique Params) ---")
            
            best_avg_metric = float('inf')
            true_best_params = None
            true_best_trial_number = -1
            
            for i, trial in enumerate(unique_trials[:top_k]):
                print(f"\nRe-evaluating Trial {trial.number} (Optuna Valid {args.metric}: {trial.value:.4f})")
                print(f"  Params: {trial.params}")
                
                data_splits_reval = []
                for ds_idx in [1, 2, 3]:
                    utils.set_seed(42)
                    ds_path = os.path.join(parent_dir, 'data', f'ihdp_semi_{ds_idx}.pkl') if args.dataset == 'ihdp' else os.path.join(parent_dir, 'data', f'cont_synthetic_{ds_idx}.pkl')
                    with open(ds_path, 'rb') as f:
                        data_k = pickle.load(f)
                    dataset_k = {k: data_k[k] for k in data_k.keys()}
                    set_train_k, set_val_k, set_test_k, coefs_k = utils.split_data(dataset_k, train_size=0.63, val_size=0.27)
                    info_k = {
                        "dim_x": data_k['x'].shape[1],
                        "dim_a": 1,
                        "dim_y": 1,
                        "HSIC_xa": utils.train_hsic(set_train_k),
                        "dep_z": utils.cached_dependence_z(set_train_k, f"{args.dataset}_{ds_idx}"),
                    }
                    t_range_k = (np.percentile(data_k['a'], 5), np.percentile(data_k['a'], 95))
                    data_splits_reval.append((set_train_k, set_val_k, set_test_k, coefs_k, info_k, t_range_k, data_k['x']))
                
                fixed_trial = optuna.trial.FixedTrial(trial.params)
                avg_metric = objective(fixed_trial, dataset_type, data_splits_reval, args.metric, args.guidance, args.selector)
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

            save_best(true_best_params)
    else:
        print("\nNo trials completed. Nothing to save.")
