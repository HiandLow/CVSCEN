import os
import json
import pickle
import numpy as np
import execute
import utils
import argparse
import matplotlib.pyplot as plt

def load_tuned(best_hparams_file):
    tune_params = {}
    if os.path.exists(best_hparams_file):
        with open(best_hparams_file, "r") as f:
            tune_params = json.load(f)
            if "coef_loss_u" in tune_params:
                tune_params["coef_loss_c"] = tune_params.pop("coef_loss_u")
            if "coef_loss_v" in tune_params:
                tune_params["coef_loss_p"] = tune_params.pop("coef_loss_v")
            if "coef_loss" in tune_params:
                tune_params["coef_loss_c"] = tune_params["coef_loss_p"] = tune_params.pop("coef_loss")
    return tune_params

def run_ablation(dataset_type, case_name, hparams_override, hparams_file=None):
    main_file = os.path.join("tune", f"best_hparams_CVSCEN_{dataset_type}.json")
    if hparams_file is not None and not os.path.exists(hparams_file):
        print(f"[{case_name}] {hparams_file} not found; using {main_file} (not separately tuned)")
        hparams_file = None
    tune_params = load_tuned(hparams_file or main_file)
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
        utils.set_seed(i)
        data_name = f'./data/ihdp_semi_{i}.pkl' if dataset_type == 'ihdp' else f'./data/cont_synthetic_{i}.pkl'
        with open(data_name, 'rb') as file:
            data = pickle.load(file)
        data['_idx'] = i

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

    main_params = load_tuned(os.path.join("tune", f"best_hparams_CVSCEN_{dataset_type}.json"))
    tuned_file = lambda suffix: os.path.join("tune", f"best_hparams_CVSCEN_{dataset_type}_{suffix}.json")
    cases = [
        {"name": "w/o Variable Selection", "label": "w/o Var Sel.", "params": {"disable_vs": True}},
        {"name": "w/ Simple MLP", "label": "w/ MLP", "params": {"use_mlp": True}},
        {"name": "Oracle", "label": "Oracle", "params": {"use_oracle": True}},
    ]
    if main_params.get("selector", "role") == "two_gate":
        cases += [
            {"name": "w/o Treatment Head", "label": "w/o Treat. Head", "params": {"weight_treat": 0.0}},
            {"name": "Role Selector", "label": "Role sel.", "params": {"selector": "role"}, "hparams_file": tuned_file("role")},
        ]
    elif main_params.get("guidance") == "calibrated":
        cases += [
            {"name": "w/o Dependence Costs", "label": "w/o Dep. Costs", "params": {"weight_dep": 0.0}},
            {"name": "w/o Sparsity", "label": "w/o Sparsity", "params": {"coef_loss_c": 0.0, "coef_loss_p": 0.0}},
            {"name": "Logit Guidance", "label": "Logit guid.", "params": {"guidance": "logit"}, "hparams_file": tuned_file("logit")},
        ]
    else:
        other_guidance = "logit" if main_params.get("guidance", "logit") == "penalty" else "penalty"
        cases += [
            {"name": "w/o HSIC Loss", "label": "w/o HSIC Loss", "params": {"weight_hsic": 0.0}},
            {"name": "w/o Correlation Prior", "label": "w/o Corr. Prior", "params": {"weight_corr": 0.0}},
            {"name": f"{other_guidance.capitalize()} Guidance", "label": f"{other_guidance.capitalize()} guid.",
             "params": {"guidance": other_guidance}, "hparams_file": tuned_file(other_guidance)},
        ]
    cases.append({"name": "Full Model", "label": "Full CVSCEN", "params": {}})

    cache_path = f"figures/ablation_cache_{dataset_type}.json"
    os.makedirs("figures", exist_ok=True)
    
    results = []
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            results = json.load(f)
            
    completed_cases = [res['case'] for res in results]

    for case in cases:
        if case['name'] in completed_cases:
            print(f"Skipping Case: {case['name']} (already completed)")
            continue
            
        print(f"Running Case: {case['name']} on {dataset_type} ...")
        res = run_ablation(dataset_type, case['name'], case['params'], case.get('hparams_file'))
        results.append(res)
        
        with open(cache_path, "w") as f:
            json.dump(results, f)
            
    output_path = f"figures/ablation_result_{dataset_type}.txt"
    
    active_case_names = [c['name'] for c in cases]
    results = [r for r in results if r['case'] in active_case_names]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Ablation Study Results on {dataset_type.upper()} Dataset\n")
        f.write("="*60 + "\n")
        for r in results:
            f.write(f"[{r['case']}]\n")
            f.write(f"MISE: {r['mise'][0]:.4f} ± {r['mise'][1]:.4f}\n")
            f.write(f"ADRFE: {r['adrfe'][0]:.4f} ± {r['adrfe'][1]:.4f}\n")
            f.write(f"FDR1: {r['fdr1'][0]:.4f} ± {r['fdr1'][1]:.4f}, FDR2: {r['fdr2'][0]:.4f} ± {r['fdr2'][1]:.4f}, FDR3: {r['fdr3'][0]:.4f} ± {r['fdr3'][1]:.4f}\n")
            f.write(f"TPR1: {r['tpr1'][0]:.4f} ± {r['tpr1'][1]:.4f}, TPR2: {r['tpr2'][0]:.4f} ± {r['tpr2'][1]:.4f}, TPR3: {r['tpr3'][0]:.4f} ± {r['tpr3'][1]:.4f}\n")
            f.write("-" * 60 + "\n")
            
    print(f"\nAblation study complete. Results saved to {output_path}")

    plot_order = [c["name"] for c in cases]
    labels_map = {c["name"]: c["label"] for c in cases}

    ordered_res = []
    for name in plot_order:
        for res in results:
            if res['case'] == name:
                ordered_res.append(res)
                break
                
    models_labels = [labels_map[name] for name in plot_order]
    
    mise_mean = [r['mise'][0] for r in ordered_res]
    mise_std = [r['mise'][1] for r in ordered_res]
    
    adrfe_mean = [r['adrfe'][0] for r in ordered_res]
    adrfe_std = [r['adrfe'][1] for r in ordered_res]
    
    fdr_mean = [[r['fdr1'][0] for r in ordered_res], [r['fdr2'][0] for r in ordered_res], [r['fdr3'][0] for r in ordered_res]]
    fdr_std = [[r['fdr1'][1] for r in ordered_res], [r['fdr2'][1] for r in ordered_res], [r['fdr3'][1] for r in ordered_res]]
    
    tpr_mean = [[r['tpr1'][0] for r in ordered_res], [r['tpr2'][0] for r in ordered_res], [r['tpr3'][0] for r in ordered_res]]
    tpr_std = [[r['tpr1'][1] for r in ordered_res], [r['tpr2'][1] for r in ordered_res], [r['tpr3'][1] for r in ordered_res]]
    
    plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    x = np.arange(len(models_labels))
    
    def plot_single(ax, means, stds, title, ylabel):
        ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=5, capthick=2, markersize=8, color='#1f77b4', ecolor='#1f77b4', linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(models_labels, rotation=30, ha='right')
        ax.set_title(title, fontsize=18, loc='left', fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlim(-0.5, len(models_labels)-0.5)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f7f7f7')
        
    def plot_multi(ax, mean_list, std_list, leg_labels, title, ylabel):
        colors = ['#2ca02c', '#ff7f0e', '#d62728']
        markers = ['o', 's', '^']
        offsets = [-0.15, 0.0, 0.15]
        for i in range(3):
            ax.errorbar(x + offsets[i], mean_list[i], yerr=std_list[i], fmt=f'{markers[i]}-', 
                        capsize=4, capthick=1.5, markersize=7, 
                        color=colors[i], ecolor=colors[i], linewidth=2, label=leg_labels[i])
        ax.set_xticks(x)
        ax.set_xticklabels(models_labels, rotation=30, ha='right')
        ax.set_title(title, fontsize=18, loc='left', fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_xlim(-0.5, len(models_labels)-0.5)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f7f7f7')
        ax.legend(loc='best', fontsize=14)
        
    plot_single(axs[0, 0], mise_mean, mise_std, "(a) MISE", "Estimation error")
    plot_single(axs[0, 1], adrfe_mean, adrfe_std, "(b) ADRF Error", "Estimation error")
    
    labels = ["Confounders", "Outcome Predictors", "Union"]
    plot_multi(axs[1, 0], fdr_mean, fdr_std, labels, "(c) FDR", "False discovery rate")
    plot_multi(axs[1, 1], tpr_mean, tpr_std, labels, "(d) TPR", "True positive rate")
    
    plt.tight_layout()
    plt.savefig(f"figures/ablation_{dataset_type}.png", dpi=300, bbox_inches='tight')
    print(f"Saved figures/ablation_{dataset_type}.png")
