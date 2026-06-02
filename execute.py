import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import argparse

import model
import utils

def evaluate_model(data, dataset_type='ihdp', tune=None):
    if dataset_type == 'ihdp':
        hparams = {
            "coef_loss": [10.0, 0.25, 0.25, 0.25],
            "weight_corr": 50.0,
            "temp_end": 0.1,
            "lr_s": 1e-2,
            "lr_p": 1e-4,
            "dim_layer": 64,
            "lambda_": 1.0,
            "epoch_total": 500,
            "step_size": 100,
            "batch_size": 32,
            "weight_decay": 1e-3,
            "L_pe": 3,
        }
    else:
        hparams = {
            "coef_loss": [1.0, 10.0, 1.15, 0.75],
            "weight_corr": 0.25,
            "temp_end": 0.1,
            "lr_s": 1e-2,
            "lr_p": 1e-4,
            "dim_layer": 256,
            "lambda_": 1.5,
            "epoch_total": 300,
            "step_size": 100,
            "batch_size": 32,
            "weight_decay": 1e-3,
            "L_pe": 3,
        }

    if tune:
        hparams.update(tune)

    dim_l = hparams["dim_layer"]
    fix = {
        "dim_fe_1": dim_l,
        "dim_fe_2": dim_l,       
        "dim_ce_1": dim_l, 
        "dim_apt_1": dim_l if dataset_type == 'ihdp' else dim_l // 2,
        "dim_apt_2": dim_l if dataset_type == 'ihdp' else dim_l // 2, 
        "temp_start": 10.0,
        "gamma": 0.97,
    }

    hparams.update(fix)

    dataset = {k: data[k] for k in data.keys()}
    set_train, set_val, set_test, coefs = utils.split_data(dataset, train_size=0.63, val_size=0.27)
    loader_train, loader_val, loader_test = [DataLoader(ds, batch_size = hparams["batch_size"], shuffle = is_train) 
        for ds, is_train in zip([set_train, set_val, set_test], [True, False, False])]

    info = {
        "dim_x": data['x'].shape[1],
        "dim_a": 1,
        "dim_y": 1,
        "HSIC_xa": coefs[2]
    }

    selector = model.VSLayer(info, hparams)
    predictor_y = model.POLayer(info, hparams)
    antipredictor_a = model.APTLayer(info, hparams)
    model_main = model.MainModel(info = {"vsl": selector, "pol": predictor_y, "aptl": antipredictor_a}, hparams = {"epoch_total": hparams["epoch_total"]})
    
    param_s = list(selector.parameters())
    param_p = list(predictor_y.parameters()) + list(antipredictor_a.parameters())

    optimizer_s = torch.optim.Adam(param_s, lr = hparams["lr_s"])
    optimizer_p = torch.optim.Adam(param_p, lr = hparams["lr_p"], weight_decay = hparams['weight_decay'])

    scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size = hparams["step_size"], gamma = hparams["gamma"])
    scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size = hparams["step_size"], gamma = hparams["gamma"])

    model_main = model.train_model(model_main, optimizer_s, optimizer_p, scheduler_s, scheduler_p, \
                                   loader_train, loader_val, hparams["coef_loss"])

    mise_tr, adrfe_tr, _, _, _, drf_tr = model.evaluate(model_main, loader_train, coefs, dataset_type)
    mise_te, adrfe_te, fdr_te, tpr_te, c_te, drf_te = model.evaluate(model_main, loader_test, coefs, dataset_type)

    return mise_tr, adrfe_tr, mise_te, adrfe_te, fdr_te, tpr_te, c_te, drf_tr, drf_te

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ihdp', choices=['ihdp', 'synt'])
    args = parser.parse_args()
    dataset_type = args.dataset

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
    c_u_te_list = np.zeros(num_features, dtype=int)
    c_v_te_list = np.zeros(num_features, dtype=int)
    c_uv_te_list = np.zeros(num_features, dtype=int)
    
    # We let evaluate_model use the default tune for the dataset type by passing tune=None
    tune = None

    for i in range(10):
        data_name = f'./data/ihdp_semi_{i}.pkl' if dataset_type == 'ihdp' else f'./data/cont_synthetic_{i}.pkl'
        with open(data_name, 'rb') as file:
            data = pickle.load(file)

        mise_tr, adrfe_tr, mise_te, adrfe_te, fdr_te, tpr_te, c_te, drf_tr, drf_te = evaluate_model(data, dataset_type, tune)
        
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
        c_u_te_list += c_te[0]
        c_v_te_list += c_te[1]
        c_uv_te_list += c_te[2]
        drf_tr_list.append(drf_tr)
        drf_te_list.append(drf_te)

        print(f"{i}th data\nIn-Sample mise: {mise_tr:.4f}, adrfe: {adrfe_tr:.4f}, \n\
Out-Sample mise: {mise_te:.4f}, adrfe: {adrfe_te:.4f} , \n\
FDR1: {fdr_te[0]:.4f}, FDR2: {fdr_te[1]:.4f} FDR3: {fdr_te[2]:.4f}, \n\
TPR1: {tpr_te[0]:.4f}, TPR2: {tpr_te[1]:.4f} TPR3: {tpr_te[2]:.4f}")
        
    avg_mise_tr = np.mean(mise_tr_list)
    avg_mise_te = np.mean(mise_te_list)

    avg_adrfe_tr = np.mean(adrfe_tr_list)
    avg_adrfe_te = np.mean(adrfe_te_list)

    avg_fdr1_te = np.mean(fdr1_te_list)
    avg_fdr2_te = np.mean(fdr2_te_list)
    avg_fdr3_te = np.mean(fdr3_te_list)

    avg_tpr1_te = np.mean(tpr1_te_list)
    avg_tpr2_te = np.mean(tpr2_te_list)
    avg_tpr3_te = np.mean(tpr3_te_list)

    avg_c_u_te = np.mean(c_u_te_list)
    avg_c_v_te = np.mean(c_v_te_list)
    avg_c_uv_te = np.mean(c_uv_te_list)

    avg_pred_tr, avg_fact_tr, t_axis_tr = np.mean(drf_tr_list, axis=0)
    avg_pred_te, avg_fact_te, t_axis_te = np.mean(drf_te_list, axis=0)

    print("Average In-Sample over all datasets and iterations\n\
mise: {:.4f} adrfe: {:.4f}".format(avg_mise_tr, avg_adrfe_tr))
    print("Average Out-Sample over all datasets and iterations\n\
mise: {:.4f} adrfe: {:.4f}".format(avg_mise_te, avg_adrfe_te))
    print("Average FDR and TPR over all datasets and iterations\n\
FDR1: {:.4f}, FDR2: {:.4f} FDR3: {:.4f}, \n\
TPR1: {:.4f}, TPR2: {:.4f} TPR3: {:.4f}".format(avg_fdr1_te, avg_fdr2_te, avg_fdr3_te, avg_tpr1_te, avg_tpr2_te, avg_tpr3_te))
    
    utils.plot_curve(data['x'].shape[1], c_u_te_list, c_v_te_list, str(np.sum(c_u_te_list)), str(np.sum(c_v_te_list)), "Each")
    utils.plot_curve(data['x'].shape[1], c_uv_te_list, np.zeros_like(c_uv_te_list), str(np.sum(c_uv_te_list)), "0", "Union")
    utils.plot_drf(t_axis_te, avg_fact_te, avg_pred_te, f"vscen_{dataset_type}")
