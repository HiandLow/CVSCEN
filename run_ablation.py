import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse

def parse_output(stdout):
    # Regex to find Average Out-Sample mise and adrfe
    mise_match = re.search(r'Average Out-Sample.*?\nmise:\s*([\d.]+)', stdout, re.DOTALL)
    adrfe_match = re.search(r'Average Out-Sample.*?\n.*?adrfe:\s*([\d.]+)', stdout, re.DOTALL)
    
    # Regex to find Average FDR and TPR
    fdr_match = re.search(r'Average FDR and TPR.*?\nFDR1:\s*([\d.]+).*?FDR2:\s*([\d.]+).*?FDR3:\s*([\d.]+)', stdout, re.DOTALL)
    tpr_match = re.search(r'TPR1:\s*([\d.]+).*?TPR2:\s*([\d.]+).*?TPR3:\s*([\d.]+)', stdout, re.DOTALL)

    return {
        'MISE': float(mise_match.group(1)) if mise_match else None,
        'ADRFE': float(adrfe_match.group(1)) if adrfe_match else None,
        'FDR1': float(fdr_match.group(1)) if fdr_match else None,
        'TPR1': float(tpr_match.group(1)) if tpr_match else None
    }

def main():
    parser = argparse.ArgumentParser(description="Run ablation combinations")
    parser.add_argument('--dataset', default='ihdp', choices=['ihdp', 'synt'])
    args = parser.parse_args()

    # Define standard combinations to run (user can modify this list or add arbitrary combinations)
    combinations = [
        {"name": "Full Model", "flags": []},
        {"name": "w/o VS", "flags": ["--no_vs"]},
        {"name": "w/o HSIC Indep", "flags": ["--no_hsic_indep"]},
        {"name": "w/o HSIC Guide", "flags": ["--no_hsic_guide"]},
        {"name": "w/o Mod", "flags": ["--no_mod"]},
        {"name": "w/o Guide & Mod", "flags": ["--no_hsic_guide", "--no_mod"]},
    ]

    results = []

    # Use the current python executable to ensure conda env is used correctly
    python_exe = sys.executable

    print(f"Starting Ablation Pipeline on dataset: {args.dataset.upper()}")
    print("-" * 50)

    for combo in combinations:
        print(f"[{combo['name']}] Running...")
        cmd = [python_exe, "execute.py", "--dataset", args.dataset] + combo["flags"]
        
        # Run process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"[!] Error running {combo['name']}:\n{stderr}")
            continue
            
        metrics = parse_output(stdout)
        metrics['Model'] = combo['name']
        results.append(metrics)
        print(f"  -> MISE: {metrics['MISE']}, FDR1: {metrics['FDR1']}, TPR1: {metrics['TPR1']}")

    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    cols = ['Model'] + [c for c in df.columns if c != 'Model']
    df = df[cols]
    
    # Save CSV
    csv_path = f"ablation_results_{args.dataset}.csv"
    df.to_csv(csv_path, index=False)
    print("\n" + "=" * 50)
    print(f"Results successfully saved to {csv_path}")

    # Plot
    os.makedirs('figures', exist_ok=True)
    df_plot = df.dropna(subset=['MISE'])
    
    if not df_plot.empty:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df_plot['Model'], df_plot['MISE'], color='#4a90e2')
        plt.ylabel('Out-Sample MISE', fontsize=12)
        plt.title(f'Ablation Study Results ({args.dataset.upper()})', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(f'figures/ablation_summary_{args.dataset}.png', dpi=300)
        print(f"Plot successfully saved to figures/ablation_summary_{args.dataset}.png")

if __name__ == "__main__":
    main()
