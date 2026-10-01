import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import ignite.metrics
from scipy.integrate import romb

def split_data(data, train_size=0.63, val_size=0.27, seed=42):
    n = data['x'].shape[0]
    indices = np.random.RandomState(seed).permutation(n)
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)
    data_keys = ['x', 'a', 'yf']
    coef_keys = ["treat_coef", "out_coef"]

    def make_dataset(idxs, is_data = True):
        if is_data:
            dataset = torch.utils.data.TensorDataset(*[torch.as_tensor(data[k][idxs], dtype = torch.float32).reshape(len(idxs), -1) for k in data_keys])

        else:
            dataset = [torch.as_tensor(data[k], dtype = torch.float32) for k in coef_keys if k in data]

        return dataset

    return make_dataset(indices[:train_end]), make_dataset(indices[train_end:val_end]), \
        make_dataset(indices[val_end:]), make_dataset(None, is_data = False)

def HSIC(x, a):
    # median heuristic over distinct pairs: ignite's per-batch median is 0 for binary/discrete x (-> NaN)
    d = np.subtract.outer(x, x).astype(np.float64) ** 2
    d = d[d > 0]
    sigma_x = float(np.sqrt(np.median(d))) if d.size else -1
    x, a = torch.tensor(x).unsqueeze(1), torch.tensor(a)
    hsic = ignite.metrics.HSIC(sigma_x=sigma_x)
    batch_size=256

    for idx in range(0, x.size(0), batch_size):
        hsic.update((x[idx : idx + batch_size], a[idx : idx + batch_size]))

    return hsic.compute()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _median_gram(v):
    d = (v[:, None] - v[None, :]) ** 2
    nz = d[d > 1e-12]
    s2 = nz.median() if nz.numel() else torch.tensor(1.0, dtype=v.dtype, device=v.device)
    return torch.exp(-d / (2 * s2))

def dependence_z(set_train, n_perm=300, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = set_train.tensors[0].to(device, torch.float64)
    a = set_train.tensors[1].to(device, torch.float64).view(-1)
    n = a.shape[0]
    h = torch.eye(n, device=device, dtype=torch.float64) - 1.0 / n
    l = h @ _median_gram(a) @ h
    gen = torch.Generator(device=device).manual_seed(seed)
    perms = [torch.randperm(n, generator=gen, device=device) for _ in range(n_perm)]
    z = []
    for j in range(x.shape[1]):
        k = _median_gram(x[:, j])
        stat = (k * l).sum()
        null = torch.stack([(k * l[p][:, p]).sum() for p in perms])
        z.append(((stat - null.mean()) / (null.std() + 1e-12)).item())
    return torch.tensor(z, dtype=torch.float32)

def cached_dependence_z(set_train, key):
    import os
    os.makedirs('cache', exist_ok=True)
    path = os.path.join('cache', f'z_{key}.pt')
    if os.path.exists(path):
        return torch.load(path)
    z = dependence_z(set_train)
    torch.save(z, path)
    return z

def train_hsic(set_train):
    x, a = set_train.tensors[0].numpy(), set_train.tensors[1].numpy().reshape(-1, 1)
    scores = [HSIC(x[:, j], a) for j in range(x.shape[1])]
    return torch.tensor(np.nan_to_num(np.array(scores, dtype=np.float64)), dtype=torch.float32)

def curve_error(pred_grid, target_grid, dx, adrf_only=False):
    if adrf_only:
        return np.sqrt(romb((target_grid.mean(axis=0) - pred_grid.mean(axis=0)) ** 2, dx=dx))
    diff_sq = (target_grid - pred_grid) ** 2
    return np.sqrt(np.mean([romb(diff_sq[i], dx=dx) for i in range(len(diff_sq))]))

def FDR(select, coef_a, coef_y, is_confounder):
    if torch.sum(select) == 0:
        return 0.0
    
    select = select != 0
    fact = ((coef_a != 0) == is_confounder) & (coef_y != 0)
    fdr = torch.sum((select == True) & (fact == False)) / torch.sum(select)

    return fdr.item()

def TPR(select, coef_a, coef_y, is_confounder):
    fact = ((coef_a != 0) == is_confounder) & (coef_y != 0)
    if torch.sum(fact) == 0:
        return 0.0
    
    select = select != 0
    true_positives = torch.sum((select == True) & (fact == True))
    tpr = true_positives / torch.sum(fact)

    return tpr.item()    

def plot_curve(axis, a, b, string_a, string_b, file_name, treat_coef=None, out_coef=None, max_noise_bars=10):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if treat_coef is None or out_coef is None:
        roles = [""] * axis
    else:
        ta = np.asarray(treat_coef).reshape(-1) != 0
        oy = np.asarray(out_coef).reshape(-1) != 0
        roles = ["C" if t and o else "P" if o else "I" if t else "N" for t, o in zip(ta, oy)]

    noise_idx = [j for j in range(axis) if roles[j] == "N"]
    group_noise = len(noise_idx) > max_noise_bars
    shown = [j for j in range(axis) if not (group_noise and roles[j] == "N")]

    x_labels = [f"{j+1}\n({roles[j]})" if roles[j] else str(j + 1) for j in shown]
    grouped_a = [a[j] for j in shown]
    grouped_b = [b[j] for j in shown]
    if group_noise:
        x_labels.append(f"N\n(avg of {len(noise_idx)})")
        grouped_a.append(a[noise_idx].mean())
        grouped_b.append(b[noise_idx].mean())

    x = np.arange(len(x_labels))
    width = 0.35

    plt.figure(figsize=(20, 6))

    plt.bar(x - width/2, grouped_a, width, label=string_a, color='#FF7675')
    plt.bar(x + width/2, grouped_b, width, label=string_b, color='#74B9FF')

    plt.xticks(x, x_labels)
    plt.tick_params(axis='x', rotation=45, labelsize=10)
    plt.xlim(-1, len(x_labels))

    if group_noise:
        plt.axvline(x=len(shown) - 0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        max_height = max(max(grouped_a), max(grouped_b), 1e-8)
        plt.text(len(shown) - 0.7, max_height * 0.9, 'Noise Features (Averaged) →', color='gray', fontsize=12, fontweight='bold', ha='right')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.savefig('./figures/' + file_name + '.png')
    plt.close()

def plot_drf(t_axis, avg_fact, avg_pred, file_name):
    model_name = file_name.split('_')[0].upper()
    if model_name == "VSCEN":
        model_name = "CVSCEN"
        
    plt.figure(figsize=(8, 6))
    plt.plot(t_axis, avg_fact, color='black', linestyle='--', linewidth=3, label='True ADRF')
    plt.plot(t_axis, avg_pred, color='red', linewidth=3, label='Pred ADRF')
    plt.fill_between(t_axis, avg_fact, avg_pred, color='red', alpha=0.1)
    
    plt.title(model_name, fontsize=24)
    plt.xlabel("Treatment (a)", fontsize=20)
    plt.ylabel("Outcome (Y)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/' + file_name +'.png')
    plt.close()

if __name__ == "__main__":
    test = FDR(torch.tensor([0, 0, 0]), torch.tensor([0.5, 0.0, 0.5]), torch.tensor([0.5, 0.5, 0.5]), is_confounder=True)
    print(test)