import argparse
import numpy as np
import pandas as pd
import pickle
import os

# IHDP columns (data/ihdp.csv[:, 2:27]):
#  0 bw, 1 b.head, 2 preterm, 3 birth.o, 4 nnhealth, 5 momage, 6 sex, 7 twin, 8 b.marr,
#  9 mom.lths, 10 mom.hs, 11 mom.scoll, 12 cig, 13 first, 14 booze, 15 drugs, 16 work.dur,
#  17 prenatal, 18-24 sites (ark, ein, har, mia, pen, tex, was)
# Treatment a = intervention intensity, outcome y = cognitive score.
IHDP_CONT = [0, 1, 2, 3, 4, 5]
IHDP_C = [0, 1, 2, 4, 5, 8, 9]           # infant health risk + maternal SES (mom.lths)
IHDP_P = [3, 6, 7, 12, 14]               # child / pregnancy factors
IHDP_W = [18, 19, 20, 21, 22, 23, 24]    # sites
IHDP_N = [10, 11, 13, 15, 16, 17]        # mom.hs, mom.scoll, first (= birth.o==1), drugs, work.dur, prenatal

# Effects in standardised units (every covariate, binary included, is z-scored inside the DGP),
# so a coefficient adds the same variance regardless of prevalence. Linear terms are listed here;
# bw (threshold), nnhealth (saturating) and momage (V-shape) enter nonlinearly below.
IHDP_TREAT_C = {1: -0.25, 2: 0.30, 8: -0.35, 9: 0.45}
IHDP_TREAT_W = {18: 0.30, 19: -0.25, 20: 0.25, 21: -0.30, 22: 0.30, 23: -0.25, 24: 0.25}
IHDP_OUT_C = {1: 0.30, 2: -0.35, 8: 0.30, 9: -0.40}
IHDP_OUT_P = {3: -0.35, 6: 0.30, 7: -0.35, 12: -0.35, 14: -0.35}

def preprocess_ihdp(X_raw):
    X = X_raw.astype(np.float64).copy()
    X[:, 13] -= 1.  # first: {1, 2} -> {0, 1}, 1 = firstborn
    X[:, IHDP_CONT] = (X[:, IHDP_CONT] - X[:, IHDP_CONT].mean(axis=0)) / X[:, IHDP_CONT].std(axis=0)
    return X

def _ihdp_z(X, X_ref):
    X = np.atleast_2d(X)
    ref = X if X_ref is None else np.atleast_2d(X_ref)
    return (X - ref.mean(axis=0)) / ref.std(axis=0)

def ihdp_treat_logit(X, X_ref=None):
    Z = _ihdp_z(X, X_ref)
    l_c = (sum(b * Z[:, j] for j, b in IHDP_TREAT_C.items())
           + 0.45 * np.tanh(-1.5 * Z[:, 0])       # low birth weight: effect saturates away from the threshold
           + 0.45 * np.tanh(-Z[:, 4])             # poor neonatal health
           + 0.30 * (np.abs(Z[:, 5]) - 0.8))      # teenage and older mothers (V-shape in momage)
    l_w = sum(b * Z[:, j] for j, b in IHDP_TREAT_W.items())
    return l_c, l_w

def get_effect_ihdp(X, t, coefs=None, X_ref=None):
    Z = _ihdp_z(X, X_ref)
    mu = (sum(b * Z[:, j] for j, b in (IHDP_OUT_C | IHDP_OUT_P).items())
          + 0.45 * np.tanh(1.5 * Z[:, 0])
          + 0.45 * np.tanh(Z[:, 4])
          - 0.30 * (np.abs(Z[:, 5]) - 0.8))
    # frailer infants peak at a higher intensity
    risk = (-Z[:, 0] - Z[:, 1] + Z[:, 2] - Z[:, 4]) / 2.
    a_star = 0.45 + 0.15 * np.tanh(risk)
    # benefit size varies strongly across children (heavier LBW infants benefit more, as in the original IHDP
    # findings) and can turn negative. The index is additive, every term has the same sign as in mu (so its effect
    # is not cancelled), and the dose curve is non-negative so covariate effects do not cancel across doses.
    s = 0.4 * Z[:, 0] + 0.3 * Z[:, 6] - 0.4 * Z[:, 12] - 0.3 * Z[:, 3]
    gain = 1.5 + 2.0 * s
    r = t / a_star
    # population-level oscillation in the dose (as in the VCNet-style DGP), kept free of x so that it cannot
    # cancel any covariate's effect
    base = np.sin(3. * np.pi * t) / (1.2 - t)
    return mu + base + gain * (r * np.exp(1. - r)) ** 3

def generate_ihdp_data(X, seed=42, noise_std=0.5, treat_noise_std=0.6):
    np.random.seed(seed)
    n_samples, dim_x = X.shape

    l_c, l_w = ihdp_treat_logit(X)
    lin_t = l_c + l_w + np.random.normal(0, treat_noise_std, size=n_samples)
    nlin_A = 1. / (1. + np.exp(-lin_t))

    yf = get_effect_ihdp(X, nlin_A)
    yf += np.random.normal(0, noise_std, size=n_samples)

    dataset = {
        'x': X.astype(np.float32),
        'a': nlin_A.flatten().astype(np.float32),
        'yf': yf.flatten().astype(np.float32),
        'treat_coef': np.isin(np.arange(dim_x), IHDP_C + IHDP_W).astype(np.float32),
        'out_coef': np.isin(np.arange(dim_x), IHDP_C + IHDP_P).astype(np.float32),
    }
    return dataset

def get_effect_synt(X, a, coefs):
    base_Y = np.dot(X, coefs[1])
    coef = 4
    effect = (a**2) / (0.8**2 + a**2)
    return base_Y + coef * effect

def generate_synthetic_data(X, coefs, seed=42, noise_std=0.1):
    np.random.seed(seed)
    n_samples, dim_x = X.shape

    c_A_x, c_Y_x = coefs[0], coefs[1]
    
    A = normalize(np.dot(X, c_A_x) + np.random.normal(0, noise_std, size=(n_samples,1)))
    yf = get_effect_synt(X, A, coefs) + np.random.normal(0, noise_std, size=(n_samples,1))
    
    dataset = {
        'x': X.astype(np.float32),
        'a': A.flatten().astype(np.float32),
        'yf': yf.flatten().astype(np.float32),
        'treat_coef': c_A_x.flatten().astype(np.float32),
        'out_coef': c_Y_x.flatten().astype(np.float32),
    }
    return dataset

def true_curves(X, treat_grid, dataset_type, coefs, X_ref=None):
    X = np.asarray(X, dtype=np.float64)
    if dataset_type == 'ihdp':
        return np.column_stack([get_effect_ihdp(X, t, X_ref=X_ref) for t in treat_grid])
    out_coef = np.asarray(coefs[1], dtype=np.float64).reshape(-1)
    return np.column_stack([get_effect_synt(X, t, (None, out_coef)) for t in treat_grid])

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
        data_raw = pd.read_csv('data/ihdp.csv').to_numpy()
        X_raw = preprocess_ihdp(data_raw[:, 2:27])
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
