from torch.utils.data import DataLoader
import pickle
import numpy as np
import argparse

import model
import utils

def evaluate_model(data, dataset_type='ihdp', tune=None):
    hparams = {
        "selector": "role",
        "weight_hsic": 1.0,
        "coef_loss_c": 0.1,
        "coef_loss_p": 0.1,
        "weight_corr": 5.0,
        "weight_treat": 1.0,
        "coef_loss_y": 0.1,
        "coef_loss_a": 0.1,
        "init_logit": 1.0,
        "lr_s": 1e-2,
        "lr_p": 1e-3,
        "dim_layer": 64 if dataset_type == 'ihdp' else 128,
        "epoch_total": 300,
    }

    if tune:
        hparams.update(tune)

    bs = hparams.get("batch_size", 64)

    dataset = {k: data[k] for k in data.keys()}
    set_train, set_val, set_test, coefs = utils.split_data(dataset, train_size=0.63, val_size=0.27)
    loader_train, loader_val, loader_test = [DataLoader(ds, batch_size=bs, shuffle=is_train)
        for ds, is_train in zip([set_train, set_val, set_test], [True, False, False])]

    coef_a, coef_y = coefs[0], coefs[1]
    info = {
        "dim_x": data['x'].shape[1],
        "dim_a": 1,
        "dim_y": 1,
        "oracle_c": ((coef_a != 0) & (coef_y != 0)).float(),
        "oracle_p": ((coef_a == 0) & (coef_y != 0)).float()
    }
    if hparams["selector"] == "role":
        info["HSIC_xa"] = utils.train_hsic(set_train)

    model_main, optimizer_s, optimizer_p, scheduler_s, scheduler_p, coef_loss = model.build(info, hparams)
    model_main, _, _ = model.train_model(model_main, optimizer_s, optimizer_p, scheduler_s, scheduler_p,
                                         loader_train, loader_val, coef_loss)

    t_range = (np.percentile(data['a'], 5), np.percentile(data['a'], 95))
    mise_tr, adrfe_tr, _, _, _, drf_tr = model.evaluate(model_main, loader_train, coefs, dataset_type, t_range=t_range, x_ref=data['x'])
    mise_te, adrfe_te, fdr_te, tpr_te, c_te, drf_te = model.evaluate(model_main, loader_test, coefs, dataset_type, t_range=t_range, x_ref=data['x'])

    return mise_tr, adrfe_tr, mise_te, adrfe_te, fdr_te, tpr_te, c_te, drf_tr, drf_te

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    parser.add_argument('--selector', type=str, default=None, choices=['role', 'two_gate'],
                        help="Override the selector; by default it comes from the tuned file (else 'role')")
    parser.add_argument('--guidance', type=str, default=None, choices=['penalty', 'logit'],
                        help="Override the guidance of the role selector")
    parser.add_argument('--hparams', type=str, default=None,
                        help="Tuned hyperparameter JSON to load (default: tune/best_hparams_CVSCEN_{dataset}[_two_gate].json)")
    args = parser.parse_args()
    dataset_type = args.dataset

    import os
    import json
    if args.hparams:
        candidates = [args.hparams]
    elif args.selector == 'two_gate':
        candidates = [os.path.join("tune", f"best_hparams_CVSCEN_{dataset_type}_two_gate.json"),
                      os.path.join("tune", f"best_hparams_CVSCEN_{dataset_type}.json")]
    else:
        candidates = [os.path.join("tune", f"best_hparams_CVSCEN_{dataset_type}.json")]
    best_hparams_file = next((c for c in candidates if os.path.exists(c)), None)

    tune_params = {}
    if best_hparams_file:
        print(f"Loading tuned hyperparameters from {best_hparams_file}...")
        with open(best_hparams_file, "r") as f:
            tune_params = json.load(f)
            if "coef_loss_u" in tune_params:
                tune_params["coef_loss_c"] = tune_params.pop("coef_loss_u")
            if "coef_loss_v" in tune_params:
                tune_params["coef_loss_p"] = tune_params.pop("coef_loss_v")
    elif args.hparams:
        raise FileNotFoundError(args.hparams)
    else:
        print("No tuned hyperparameter file found; using the defaults in evaluate_model.")

    tuned_selector = tune_params.get("selector")
    if args.selector:
        tune_params["selector"] = args.selector
    if args.guidance:
        tune_params["guidance"] = args.guidance
    selector = tune_params.get("selector", "role")
    if tuned_selector is not None and selector != tuned_selector:
        print(f"Note: {best_hparams_file} was tuned for selector '{tuned_selector}'; "
              f"'{selector}'-specific settings fall back to the defaults in evaluate_model.")
    print(f"Selector: {selector}" + (f", guidance: {tune_params.get('guidance', 'penalty')}" if selector == 'role' else ""))
    run_tag = "" if selector == 'role' else f"_{selector}"

    mise_tr_list = []
    mise_te_list = []
    adrfe_tr_list = []
    adrfe_te_list = []
    fdr1_te_list = []
    fdr2_te_list = []
    fdr3_te_list = []
    tpr1_te_list = []
    tpr2_te_list = []
    tpr3_te_list = []
    drf_tr_list = []
    drf_te_list = []

    num_features = 25 if dataset_type == 'ihdp' else 100
    c_c_te_list = np.zeros(num_features, dtype=int)
    c_p_te_list = np.zeros(num_features, dtype=int)
    c_cp_te_list = np.zeros(num_features, dtype=int)

    for i in range(10):
        utils.set_seed(i)
        data_name = f'./data/ihdp_semi_{i}.pkl' if dataset_type == 'ihdp' else f'./data/cont_synthetic_{i}.pkl'
        with open(data_name, 'rb') as file:
            data = pickle.load(file)

        mise_tr, adrfe_tr, mise_te, adrfe_te, fdr_te, tpr_te, c_te, drf_tr, drf_te = evaluate_model(data, dataset_type, tune_params)
        
        mise_tr_list.append(mise_tr)
        mise_te_list.append(mise_te)
        adrfe_tr_list.append(adrfe_tr)
        adrfe_te_list.append(adrfe_te)
        fdr1_te_list.append(fdr_te[0])
        fdr2_te_list.append(fdr_te[1])
        fdr3_te_list.append(fdr_te[2])
        tpr1_te_list.append(tpr_te[0])
        tpr2_te_list.append(tpr_te[1])
        tpr3_te_list.append(tpr_te[2])
        c_c_te_list += c_te[0]
        c_p_te_list += c_te[1]
        c_cp_te_list += c_te[2]
        drf_tr_list.append(drf_tr)
        drf_te_list.append(drf_te)

        print(f"{i}th data\nIn-Sample mise: {mise_tr:.4f}, adrfe: {adrfe_tr:.4f}, \n\
Out-Sample mise: {mise_te:.4f}, adrfe: {adrfe_te:.4f} , \n\
FDR1: {fdr_te[0]:.4f}, FDR2: {fdr_te[1]:.4f} FDR3: {fdr_te[2]:.4f}, \n\
TPR1: {tpr_te[0]:.4f}, TPR2: {tpr_te[1]:.4f} TPR3: {tpr_te[2]:.4f}")
        
    avg_mise_tr, std_mise_tr = np.mean(mise_tr_list), np.std(mise_tr_list)
    avg_adrfe_tr, std_adrfe_tr = np.mean(adrfe_tr_list), np.std(adrfe_tr_list)
    avg_mise_te, std_mise_te = np.mean(mise_te_list), np.std(mise_te_list)
    avg_adrfe_te, std_adrfe_te = np.mean(adrfe_te_list), np.std(adrfe_te_list)

    avg_fdr1_te, std_fdr1_te = np.mean(fdr1_te_list), np.std(fdr1_te_list)
    avg_fdr2_te, std_fdr2_te = np.mean(fdr2_te_list), np.std(fdr2_te_list)
    avg_fdr3_te, std_fdr3_te = np.mean(fdr3_te_list), np.std(fdr3_te_list)

    avg_tpr1_te, std_tpr1_te = np.mean(tpr1_te_list), np.std(tpr1_te_list)
    avg_tpr2_te, std_tpr2_te = np.mean(tpr2_te_list), np.std(tpr2_te_list)
    avg_tpr3_te, std_tpr3_te = np.mean(tpr3_te_list), np.std(tpr3_te_list)

    avg_c_c_te = np.mean(c_c_te_list)
    avg_c_p_te = np.mean(c_p_te_list)
    avg_c_cp_te = np.mean(c_cp_te_list)

    avg_pred_tr, avg_fact_tr, t_axis_tr = np.mean(drf_tr_list, axis=0)
    avg_pred_te, avg_fact_te, t_axis_te = np.mean(drf_te_list, axis=0)

    print("Average In-Sample over all datasets and iterations\n\
mise: {:.4f} ± {:.4f} adrfe: {:.4f} ± {:.4f}".format(avg_mise_tr, std_mise_tr, avg_adrfe_tr, std_adrfe_tr))
    print("Average Out-Sample over all datasets and iterations\n\
mise: {:.4f} ± {:.4f} adrfe: {:.4f} ± {:.4f}".format(avg_mise_te, std_mise_te, avg_adrfe_te, std_adrfe_te))
    print("Average FDR and TPR over all datasets and iterations\n\
FDR1: {:.4f} ± {:.4f}, FDR2: {:.4f} ± {:.4f} FDR3: {:.4f} ± {:.4f}, \n\
TPR1: {:.4f} ± {:.4f}, TPR2: {:.4f} ± {:.4f} TPR3: {:.4f} ± {:.4f}".format(
        avg_fdr1_te, std_fdr1_te, avg_fdr2_te, std_fdr2_te, avg_fdr3_te, std_fdr3_te, 
        avg_tpr1_te, std_tpr1_te, avg_tpr2_te, std_tpr2_te, avg_tpr3_te, std_tpr3_te))
    
    utils.plot_curve(data['x'].shape[1], c_c_te_list, c_p_te_list, str(np.sum(c_c_te_list)), str(np.sum(c_p_te_list)), "Each" + run_tag)
    utils.plot_curve(data['x'].shape[1], c_cp_te_list, np.zeros_like(c_cp_te_list), str(np.sum(c_cp_te_list)), "0", "Union" + run_tag)
    utils.plot_drf(t_axis_te, avg_fact_te, avg_pred_te, f"cvscen_{dataset_type}{run_tag}")
