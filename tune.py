import optuna
import numpy as np
import execute
import pickle
import argparse

def objective_ihdp(trial):
    tune = {
        "coef_loss": [
            trial.suggest_float("c_y", 1.0, 50.0),
            trial.suggest_float("c_anti", 0.1, 10.0),
            trial.suggest_float("c_u", 0.1, 5.0),
            trial.suggest_float("c_v", 0.1, 5.0),
        ],
        "weight_corr": trial.suggest_float("weight_corr", 5.0, 200.0),
        "temp_end": trial.suggest_float("temp_end", 0.01, 0.2, log=True),
        "lr_s": trial.suggest_float("lr_s", 1e-4, 1e-2, log=True),
        "lr_p": trial.suggest_float("lr_p", 1e-5, 1e-3, log=True),
        "dim_layer": trial.suggest_categorical("dim_layer", [32, 64, 128, 256]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epoch_total": 500,
    }

    score_list = []
    for i in range(3): 
        try:
            with open(f'./data/ihdp_semi_{i}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            results = execute.evaluate_model(data, dataset_type='ihdp', tune=tune)
            # results[5] is tpr_te, results[4] is fdr_te
            score_list.append(results[5][1] - results[4][1]) 
        except FileNotFoundError:
            continue

    return np.mean(score_list) if score_list else float('-inf')

def objective_synt(trial):
    hparams = {
        "coef_loss": [
            1.0, 
            trial.suggest_float("coef_loss_anti", 0.1, 5.0),
            trial.suggest_float("coef_loss_sparse_u", 1.0, 5.0),
            trial.suggest_float("coef_loss_sparse_v", 1.0, 5.0)
        ],
        "weight_corr": trial.suggest_float("weight_corr", 0.1, 5.0),
        "temp_end": trial.suggest_float("temp_end", 0.001, 1.0, log=True),
        "lr_s": trial.suggest_float("lr_s", 1e-4, 1e-1, log=True),
        "lr_p": trial.suggest_float("lr_p", 1e-5, 1e-2, log=True),
        "dim_layer": trial.suggest_categorical("dim_layer", [128, 256, 512]),
        "lambda_": trial.suggest_float("lambda_", 0.1, 10.0),
        "epoch_total": 300,
        "batch_size": 32,
        "weight_decay": 1e-3,
    }

    result = []
    try:
        with open('./data/cont_synthetic_0.pkl', 'rb') as file:
            data = pickle.load(file)
        results = execute.evaluate_model(data, dataset_type='synt', tune=hparams)
        result.append(results)
    except FileNotFoundError:
        return float('inf'), float('inf')

    # results[2] is mise_te, results[4] is fdr_te
    mise_avg = np.mean([res[2] for res in result])
    fdr_c_avg = np.mean([res[4][0] for res in result])
    fdr_p_avg = np.mean([res[4][-1] for res in result])

    return mise_avg, 0.7 * fdr_c_avg + 0.3 * fdr_p_avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    args = parser.parse_args()
    
    if args.dataset == 'ihdp':
        storage_url = "sqlite:///optuna_study.db"
        study = optuna.create_study(
            study_name="cvscen_optimization",
            direction="maximize",
            storage=storage_url,
            load_if_exists=True
        )
        study.optimize(objective_ihdp, n_trials=5000)
        print(f"\n[최적화 완료]")
        print(f"Best Score (TPR-FDR): {study.best_value:.4f}")
        print("Best Hyperparameters:")
        print(study.best_params)
    else:
        study = optuna.create_study(
            study_name="vscen_tuning_cont", 
            storage="sqlite:///vscen_tuning_synt.db", 
            load_if_exists=True, 
            directions=["minimize", "minimize"]
        )
        study.optimize(objective_synt, n_trials=500)
        print("Best trials:")
        for t in study.best_trials:
            print(f"  Values: {t.values}, Params: {t.params}")
