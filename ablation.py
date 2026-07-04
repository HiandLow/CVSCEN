import os
import json
import pickle
import numpy as np
import execute
import argparse

def run_ablation(dataset_type, case_name, hparams_override):
    # Load tuned hparams
    tune_params = {}
    best_hparams_file = os.path.join("tune", f"best_hparams_CVSCEN_{dataset_type}.json")
    if os.path.exists(best_hparams_file):
        with open(best_hparams_file, "r") as f:
            tune_params = json.load(f)
            # Backward compatibility
            if "coef_loss_u" in tune_params:
                tune_params["coef_loss_c"] = tune_params.pop("coef_loss_u")
            if "coef_loss_v" in tune_params:
                tune_params["coef_loss_p"] = tune_params.pop("coef_loss_v")
                
    # Override with ablation parameters
    tune_params.update(hparams_override)
    
    mise_te_list = []
    adrfe_te_list = []
    fdr1_list = []
    fdr2_list = []
    fdr3_list = []
    tpr1_list = []
    tpr2_list = []
    tpr3_list = []
    
    for i in range(10):
        data_name = f'./data/ihdp_semi_{i}.pkl' if dataset_type == 'ihdp' else f'./data/cont_synthetic_{i}.pkl'
        with open(data_name, 'rb') as file:
            data = pickle.load(file)
            
        _, _, mise_te, adrfe_te, fdr_te, tpr_te, _, _, _ = execute.evaluate_model(data, dataset_type, tune_params)
        
        mise_te_list.append(mise_te)
        adrfe_te_list.append(adrfe_te)
        fdr1_list.append(fdr_te[0])
        fdr2_list.append(fdr_te[1])
        fdr3_list.append(fdr_te[2])
        tpr1_list.append(tpr_te[0])
        tpr2_list.append(tpr_te[1])
        tpr3_list.append(tpr_te[2])
        
    return {
        "case": case_name,
        "mise": (np.mean(mise_te_list), np.std(mise_te_list)),
        "adrfe": (np.mean(adrfe_te_list), np.std(adrfe_te_list)),
        "fdr1": (np.mean(fdr1_list), np.std(fdr1_list)),
        "fdr2": (np.mean(fdr2_list), np.std(fdr2_list)),
        "fdr3": (np.mean(fdr3_list), np.std(fdr3_list)),
        "tpr1": (np.mean(tpr1_list), np.std(tpr1_list)),
        "tpr2": (np.mean(tpr2_list), np.std(tpr2_list)),
        "tpr3": (np.mean(tpr3_list), np.std(tpr3_list)),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='synt', choices=['ihdp', 'synt'])
    args = parser.parse_args()
    dataset_type = args.dataset

    cases = [
        {"name": "w/o HSIC Loss", "params": {"weight_hsic": 0.0}},
        {"name": "w/o Correlation Prior", "params": {"weight_corr": 0.0}},
        {"name": "w/o Both", "params": {"weight_hsic": 0.0, "weight_corr": 0.0}},
        {"name": "Full Model", "params": {}},
    ]
    
    results = []
    for case in cases:
        print(f"Running Case: {case['name']} on {dataset_type} ...")
        res = run_ablation(dataset_type, case['name'], case['params'])
        results.append(res)
        
    output_path = f"figures/ablation_result_{dataset_type}.txt"
    os.makedirs("figures", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Ablation Study Results on {dataset_type.upper()} Dataset\n")
        f.write("="*60 + "\n")
        for res in results:
            f.write(f"[{res['case']}]\n")
            f.write(f"MISE: {res['mise'][0]:.4f} ± {res['mise'][1]:.4f}\n")
            f.write(f"ADRFE: {res['adrfe'][0]:.4f} ± {res['adrfe'][1]:.4f}\n")
            f.write(f"FDR1: {res['fdr1'][0]:.4f} ± {res['fdr1'][1]:.4f}, FDR2: {res['fdr2'][0]:.4f} ± {res['fdr2'][1]:.4f}, FDR3: {res['fdr3'][0]:.4f} ± {res['fdr3'][1]:.4f}\n")
            f.write(f"TPR1: {res['tpr1'][0]:.4f} ± {res['tpr1'][1]:.4f}, TPR2: {res['tpr2'][0]:.4f} ± {res['tpr2'][1]:.4f}, TPR3: {res['tpr3'][0]:.4f} ± {res['tpr3'][1]:.4f}\n")
            f.write("-" * 60 + "\n")
            
    print(f"\nAblation study complete. Results saved to {output_path}")
