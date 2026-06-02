import argparse
import numpy as np
import pandas as pd
import pickle
import utils
import os

# === IHDP Methods ===
def get_effect_ihdp(X, t, coefs=None, alpha=5.0, factor1=1.5, factor2=0.5):
    X = np.atleast_2d(X)
    cate_idx1 = [3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    cate_mean1 = np.mean(X[:, cate_idx1], axis=1).mean()
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5]

    y = (1. / (1.2 - t)
         * np.sin(t * 3. * np.pi)
         * (factor1 * np.tanh((X[:, cate_idx1].sum(axis=1) / 10. - cate_mean1) * alpha)
            + factor2 * np.exp(0.2 * (x1 - x5)) / (0.1 + np.minimum(np.minimum(x2, x3), x4))))
    return y

def generate_ihdp_data(X, seed=42, noise_std=0.5):
    np.random.seed(seed)
    n_samples, dim_x = X.shape
    hsic = []
    alpha = 5.0
    cate_idx1 = [3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    cate_idx2 = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    factor1, factor2 = 1.5, 0.5
    cate_mean1 = np.mean(X[:, cate_idx1], axis=1).mean()
    cate_mean2 = np.mean(X[:, cate_idx2], axis=1).mean()
    x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5]

    lin_t = (x1 / (1. + x2)
             + np.maximum(np.maximum(x3, x4), x5) / (0.2 + np.minimum(np.minimum(x3, x4), x5))
             + np.tanh((X[:, cate_idx2].sum(axis=1) / 10. - cate_mean2) * alpha)
             - 2.)
    lin_t += np.random.normal(0, noise_std, size=n_samples)
    nlin_A = 1. / (1. + np.exp(-2. * lin_t))
    
    yf = get_effect_ihdp(X, nlin_A)
    yf += np.random.normal(0, noise_std, size=n_samples)

    for i in range(dim_x):
        score = utils.HSIC(X[:, i], nlin_A.reshape(-1, 1))
        if np.isnan(score):
            score = 0.0
        hsic.append(score)

    dataset = {
        'x': X.astype(np.float32),
        'a': nlin_A.flatten().astype(np.float32),
        'yf': yf.flatten().astype(np.float32),
        'treat_coef': np.isin(np.arange(dim_x), [0,1,2,4,5] + cate_idx2).astype(np.float32),
        'out_coef': np.isin(np.arange(dim_x), [0,1,2,4,5] + cate_idx1).astype(np.float32),
        'HSIC': hsic
    }
    return dataset

# === Synthetic Methods ===
def get_effect_synt(X, a, coefs):
    base_Y = np.dot(X, coefs[1])
    coef = 4
    effect = (a**2) / (0.8**2 + a**2)
    return base_Y + coef * effect

def generate_synthetic_data(X, coefs, seed=42, noise_std=0.1):
    np.random.seed(seed)
    n_samples, dim_x = X.shape
    hsic = []
    
    c_A_x, c_Y_x = coefs[0], coefs[1]
    
    A = normalize(np.dot(X, c_A_x) + np.random.normal(0, noise_std, size=(n_samples,1)))
    yf = get_effect_synt(X, A, coefs) + np.random.normal(0, noise_std, size=(n_samples,1))
    
    for i in range(dim_x):
        score = utils.HSIC(X[:, i], A)
        if np.isnan(score):
            score = 0.0
        hsic.append(score)

    dataset = {
        'x': X.astype(np.float32),
        'a': A.flatten().astype(np.float32),
        'yf': yf.flatten().astype(np.float32),
        'treat_coef': c_A_x.flatten().astype(np.float32),
        'out_coef': c_Y_x.flatten().astype(np.float32),
        'HSIC': hsic
    }
    return dataset

# === Common Methods ===
def normalize(X_raw):
    x_min = X_raw.min(axis=0)
    x_max = X_raw.max(axis=0)
    x = (X_raw - x_min) / (x_max - x_min + 1e-8)
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ihdp", "synt"])
    args = parser.parse_args()

    os.makedirs("./data", exist_ok=True)

    if args.dataset == "ihdp":
        print("Generating IHDP data...")
        data_raw = pd.read_csv('VSCEN IHDP/data/ihdp.csv').to_numpy()
        X_raw = normalize(data_raw[:, 2:27])
        for i in range(10):
            dataset = generate_ihdp_data(X_raw, seed=i, noise_std=0.5)
            with open(f"./data/ihdp_semi_{i}.pkl", "wb") as f:
                pickle.dump(dataset, f)
        print("Done IHDP.")
    elif args.dataset == "synt":
        print("Generating Synthetic data...")
        seed = 6
        np.random.seed(seed)

        n_samples, dim_x = 2000, 100
        idx_C, idx_P, idx_I, idx_N = range(0, 5), range(5, 10), range(10, 15), range(15, dim_x)
        c_A_x = np.random.uniform(0.5, 1.0, size=(dim_x, 1))
        c_Y_x = np.random.uniform(0.5, 1.0, size=(dim_x, 1))
        c_A_x[list(idx_P) + list(idx_N)] = 0
        c_Y_x[list(idx_I) + list(idx_N)] = 0
        coefs = [c_A_x, c_Y_x]

        for i in range(10):
            X_raw = np.random.normal(0, 1, (2000, 100))
            dataset = generate_synthetic_data(X_raw, coefs, seed=i)
            with open(f"./data/cont_synthetic_{i}.pkl", "wb") as f:
                pickle.dump(dataset, f)
        print("Done Synthetic.")
