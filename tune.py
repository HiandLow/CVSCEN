import optuna
import torch
from torch.utils.data import DataLoader
import pickle
import argparse
import json
import os
import sys

import model
import utils

def objective(trial, dataset_type, set_train, set_val, info):
    # Search Space
    lr_p = trial.suggest_categorical("lr_p", [1e-4, 5e-4, 1e-3, 5e-3])
    lr_s = trial.suggest_categorical("lr_s", [1e-3, 5e-3, 1e-2])
    epoch_total = trial.suggest_categorical("epoch_total", [300, 400, 500])
    dim_layer = trial.suggest_categorical("dim_layer", [32, 64, 128, 256])
    weight_hsic = trial.suggest_categorical("weight_hsic", [0.5, 1.0, 2.5, 5.0, 10.0])
    coef_loss_c = trial.suggest_categorical("coef_loss_c", [0.01, 0.05, 0.1, 0.25, 0.5])
    coef_loss_p = trial.suggest_categorical("coef_loss_p", [0.01, 0.05, 0.1, 0.25, 0.5])

    hparams = {
        "weight_hsic": weight_hsic,
        "coef_loss_c": coef_loss_c,
        "coef_loss_p": coef_loss_p,
        "weight_corr": 50.0 if dataset_type == 'ihdp' else 0.25,
        "lr_s": lr_s,
        "lr_p": lr_p,
        "dim_layer": dim_layer,
        "epoch_total": epoch_total,
    }

    # Data Loading (Moved outside objective to keep Validation set fixed across trials)
    loader_train = DataLoader(set_train, batch_size=64, shuffle=True)
    loader_val = DataLoader(set_val, batch_size=64, shuffle=False)

    # Model Initialization
    selector = model.VSLayer(info, hparams)
    predictor_y = model.POLayer(info, hparams)
    model_main = model.MainModel(info={"vsl": selector, "pol": predictor_y}, hparams={"epoch_total": hparams["epoch_total"]})
    
    param_s = list(selector.parameters())
    param_p = list(predictor_y.parameters())

    optimizer_s = torch.optim.Adam(param_s, lr=hparams["lr_s"])
    optimizer_p = torch.optim.Adam(param_p, lr=hparams["lr_p"], weight_decay=1e-3)

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
    finally:
        sys.stdout = sys.__stdout__

    # Optuna minimizes the best (minimum) validation factual MSE achieved during training
    return min(history_val_y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()
    dataset_type = args.dataset

    # Load only the first dataset block for fast tuning
    file_path = f'./data/{args.dataset}_semi_0.pkl' if args.dataset == 'ihdp' else f'./data/{args.dataset}_0.pkl'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Split data ONCE before tuning starts to ensure fair comparison
    dataset = {k: data[k] for k in data.keys()}
    set_train, set_val, set_test, coefs = utils.split_data(dataset, train_size=0.63, val_size=0.27)
    
    info = {
        "dim_x": data['x'].shape[1],
        "dim_a": 1,
        "dim_y": 1,
        "HSIC_xa": coefs[2]
    }

    print(f"Starting Optuna hyperparameter tuning for {args.dataset} dataset ({args.trials} trials)...")
    
    # Start Tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, dataset_type, set_train, set_val, info), n_trials=args.trials)

    print("\nBest Trial:")
    print("  Value (Min Valid Y_Loss): ", study.best_trial.value)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # Save to JSON
    out_file = f'best_hparams_{args.dataset}.json'
    with open(out_file, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"\nBest parameters successfully saved to {out_file}.")
