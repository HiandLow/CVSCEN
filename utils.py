import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import torch

try:
    import ignite.metrics
    import ignite
    
    class GradientReversalFunction(torch.autograd.Function):
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.clone()
    
        def backward(ctx, grads):
            lambda_ = ctx.lambda_
            lambda_ = grads.new_tensor(lambda_)
            dx = -lambda_ * grads
            return dx, None
except ImportError:
    pass
    
def split_data(data, train_size=0.63, val_size=0.27):
    n = data['x'].shape[0]
    indices = np.random.permutation(n)
    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)
    data_keys = ['x', 'a', 'yf']
    coef_keys = ["treat_coef", "out_coef", "HSIC"]

    def make_dataset(idxs, is_data = True):
        if is_data:
            dataset = torch.utils.data.TensorDataset(*[torch.as_tensor(data[k][idxs], dtype = torch.float32).reshape(len(idxs), -1) for k in data_keys])

        else:
            dataset = [torch.as_tensor(data[k], dtype = torch.float32) for k in coef_keys if k in data]

        return dataset

    return make_dataset(indices[:train_end]), make_dataset(indices[train_end:val_end]), \
        make_dataset(indices[val_end:]), make_dataset(None, is_data = False)

def HSIC(x, a):
    x, a = torch.tensor(x).unsqueeze(1), torch.tensor(a) 
    hsic = ignite.metrics.HSIC()
    batch_size=256

    for idx in range(0, x.size(0), batch_size):
        hsic.update((x[idx : idx + batch_size], a[idx : idx + batch_size]))

    return hsic.compute()

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

def plot_curve(axis, a, b, string_a, string_b, file_name, bin_size=100):
    true_a = list(a[:15])
    true_b = list(b[:15])
    x_labels = [str(i) for i in range(15)]
    
    grouped_a = true_a.copy()
    grouped_b = true_b.copy()
    
    if axis > 15:
        noise_a = a[15:]
        noise_b = b[15:]
        
        for i in range(0, len(noise_a), bin_size):
            start_idx = 15 + i
            end_idx = min(15 + i + bin_size - 1, axis - 1)
            
            x_labels.append(f"{start_idx}~\n{end_idx}")
            
            grouped_a.append(np.sum(noise_a[i:i+bin_size]))
            grouped_b.append(np.sum(noise_b[i:i+bin_size]))

    x = np.arange(len(x_labels))
    width = 0.35
    
    plt.figure(figsize=(20, 6))
    
    plt.bar(x - width/2, grouped_a, width, label=string_a, color='#FF7675')
    plt.bar(x + width/2, grouped_b, width, label=string_b, color='#74B9FF')
    
    plt.xticks(x, x_labels)
    plt.tick_params(axis='x', rotation=45, labelsize=10) 
    plt.xlim(-1, len(x_labels))
    
    if axis > 14:
        plt.axvline(x=14.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
        max_height = max(max(grouped_a), max(grouped_b))
        plt.text(14.8, max_height * 0.9, 'Noise Features (Averaged)', color='gray', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    plt.legend(fontsize=12)
    plt.savefig('./result/' + file_name + '.png')
    plt.close()

def plot_drf(t_axis, avg_fact, avg_pred, file_name):
    plt.figure(figsize=(8, 6))
    plt.plot(t_axis, avg_fact, color='black', linestyle='--', linewidth=2)
    plt.plot(t_axis, avg_pred, color='red', linewidth=2)
    plt.fill_between(t_axis, avg_fact, avg_pred, color='red', alpha=0.1)
    plt.grid(True, alpha=0.3)
    plt.savefig('./result/' + file_name +'.png')
    plt.close()

if __name__ == "__main__":
    test = FDR(torch.tensor([0, 0, 0]), torch.tensor([0.5, 0.0, 0.5]), torch.tensor([0.5, 0.5, 0.5]), is_confounder=True)
    print(test)