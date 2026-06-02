import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.special import expit
import torch
import torch.nn as nn
from src.config import SEED


def compute_mise(
    model,
    x_test,
    y_test,
    get_true_y_fn,
    source: str,
    t_dim: int,
    x_dim: int,
    n_dosage_points: int = 20,
    noise_sd: float = 0.2
) -> float:
    """
    Compute the root mean integrated squared error (MISE) over the dose-response space.

    Parameters:
        model: Callable taking (x, t) → (latent, other, y_pred)
        x_test: np.ndarray or torch.Tensor, shape (N, x_dim)
        y_test: np.ndarray or torch.Tensor, shape (N,)
        get_true_y_fn: callable, ground-truth function for y(t, x)
        source: str, either "mimic/data" (2D dosage) or "news/data" (1D dosage)
        t_dim: int, dimension of treatment vector
        x_dim: int, dimension of covariates
        n_dosage_points: int, number of discrete dosage points to integrate over
        noise_sd: float, passed to get_true_y_news (ignored for mimic)

    Returns:
        float: root MISE value
    """
    criterion = nn.MSELoss()
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    N = x_test.shape[0]

    criterion = nn.MSELoss()

    if source == "mimic/data":
        dose_a_range = np.linspace(0.0, 1.0, n_dosage_points)
        dose_b_range = np.linspace(0.0, 1.0, n_dosage_points)
        dose_combinations = np.array([[a, b] for a in dose_a_range for b in dose_b_range])
        step_size = (dose_a_range[1] - dose_a_range[0]) * (dose_b_range[1] - dose_b_range[0])
    else:
        dose_range = np.linspace(0.0, 1.0, n_dosage_points)
        dose_combinations = dose_range[:, np.newaxis]  # shape (n_dosage_points, 1)
        step_size = dose_range[1] - dose_range[0]

    mise = 0.0

    for n in range(N):
        x_n = x_test[n].unsqueeze(0)  # shape (1, x_dim)
        integral_error = 0.0

        #print(f"Current: {n/N}")

        for dose_vec in dose_combinations:
            t_input = torch.tensor(dose_vec, dtype=torch.float32).unsqueeze(0)  # shape (1, t_dim)

            # Model prediction
            _, _, y_pred = model(x_n, t_input)

            # Ground-truth outcome
            if source == "mimic/data":
                y_true_val = get_true_y_fn(t_input, x_n, dim_treat=t_dim, dim_cov=x_dim)
            else:
                y_true_val = get_true_y_fn(
                    t_input.detach().cpu().numpy(),
                    x_n.detach().cpu().numpy(),
                    noise_sd=noise_sd
                )

            y_true = torch.tensor(y_true_val, dtype=torch.float32)

            #error = criterion(y_true, y_pred)#(y_true - y_pred.detach().cpu().numpy())**2
            error = criterion(y_true,y_pred)
            integral_error += error.item()

        mise += integral_error * step_size

    mise /= N
    return np.sqrt(mise)


def get_true_y_mimic(t, x, param_interaction=2, dim_treat= 2, dim_cov = 10, noise=0.0):
    t = t.detach().numpy()
    x = x.detach().numpy()

    v = np.random.normal(loc=0.0, scale=1.0, size=(dim_treat, 2, dim_cov))
    v = v/(np.linalg.norm(v, 1, axis=2).reshape(dim_treat, 2, -1))
    
    pred_x = (np.float32(x[:, None,None,:]) * np.float32(v[None,...])).sum(3) # reducing float for big tcga dataset
    pred_x = pred_x[..., 0] / (2*pred_x[..., 1])
    pred_x = expit(pred_x)

    pred_x_adj = pred_x / 20 + 0.2

    y = 2 + 2 * (pred_x.mean(1)+0.5) * (np.cos((t - pred_x_adj)*3*np.pi) - 0.01*((t - pred_x_adj)**2)).mean(axis=1)\
        - param_interaction* 0.1*(((t-pred_x_adj)**2).prod(axis=1))
    noise = np.random.normal(0, noise, size=len(y))
    return y + noise




def compute_pe_2(
    model,
    x_test,
    get_true_y_fn,
    source: str,
    t_dim: int,
    x_dim: int,
    n_dosage_points: int = 20,
    noise_sd: float = 0.2,
) -> float:
    """
    Root Policy Error (RPE):  √(1/N · Σ (Y* - Y( t̂ ))²)

    Y*  = best-possible outcome on the dose grid (oracle)
    t̂  = dose the model would pick (argmax of its own prediction)
    """
    # ------------------------------------------------------------------ setup
    x_test = torch.as_tensor(x_test, dtype=torch.float32)
    N      = x_test.shape[0]

    # build the dose grid once ------------------------------------------------
    if source == "mimic/data":
        a = np.linspace(0.0, 1.0, n_dosage_points)
        b = np.linspace(0.0, 1.0, n_dosage_points)
        dose_combinations = np.array([[ai, bi] for ai in a for bi in b])  # (G, 2)
    else:  # "news/data"
        dose_combinations = np.linspace(0.0, 1.0, n_dosage_points)[:, None]  # (G, 1)

    t_grid = torch.as_tensor(dose_combinations, dtype=torch.float32)          # (G, t_dim)

    G      = t_grid.shape[0]

    # ------------------------------------------------------------------ loop
    sq_err_sum = 0.0
    sq_err_sum_me = 0.0

    with torch.no_grad():
        for x in x_test:                              # iterate over test points (1, x_dim)

            # replicate x to match grid size once per point -------------------
            x_rep = x.repeat(G, 1)                    # (G, x_dim)

            # model predictions on the whole grid ----------------------------
            _, _, y_pred = model(x_rep, t_grid)       # (G, 1)
            y_pred = y_pred.squeeze()                 # (G,)

            # true outcomes on the same grid ---------------------------------
            if source == "mimic/data":
                y_true = get_true_y_fn(t_grid, x_rep,
                                        dim_treat=t_dim, dim_cov=x_dim)       # (G,)
                y_true = torch.as_tensor(y_true, dtype=torch.float32)
            else:  # "news/data"
                y_true = get_true_y_fn(t_grid.cpu().numpy(),
                                        x.cpu().numpy(),
                                        noise_sd=noise_sd)                     # (G,)
                y_true = torch.as_tensor(y_true, dtype=torch.float32)

            # Retrieve the actual best true outcome and its index -------------------
            actual_best_dosage = y_true.argmax().item()  # index of the best true outcome
            outcome_under_best_actual = y_true[actual_best_dosage].item()  # true outcome at that index

            # Retrieve the predicted best true outcome and its index -------------------
            predicted_best_dosage = y_pred.argmax().item()
            outcome_under_best_predicted = y_pred[predicted_best_dosage].item()  # true outcome at that index

            sq_err_sum_me += (outcome_under_best_actual - outcome_under_best_predicted) ** 2

            # oracle vs. policy ----------------------------------------------
            best_true_val  = y_true.max().item()            # oracle outcome
            best_pred_idx  = y_pred.argmax().item()         # model-optimal dose index
            y_true_at_pred = y_true[best_pred_idx].item()   # true outcome @ that dose

            sq_err_sum += (best_true_val - y_true_at_pred) ** 2

    return np.sqrt(sq_err_sum_me/ N)


def get_true_y_news(t: np.ndarray,
               x: np.ndarray,
               noise_sd: float = 0.2,
               eps: float = 1e-8) -> np.ndarray:
    """
    y = 10 · (v₁·x + sin(π · ratio · t)) + ε,    ε ~ 𝒩(0, noise_sd²)
    ratio = (v₂·x)/(v₃·x); ratio→0 if denominator tiny.
    """
    rng          = np.random.default_rng(SEED)
    try:
        v1p, v2p, v3p = rng.normal(0.0, 1.0, size=(3, x.shape[1]))
    except:
        v1p, v2p, v3p = rng.normal(0.0, 1.0, size=(3, x.shape[0]))
        
    v1, v2, v3    = [v / np.linalg.norm(v, 2) for v in (v1p, v2p, v3p)]
    v1, v2, v3 = v1.astype(np.float32), v2.astype(np.float32), v3.astype(np.float32)

    denom = x @ v3
    ratio = np.where(np.abs(denom) < eps, 0.0, (x @ v2) / denom)
    core  = (x @ v1) + np.sin(np.pi * ratio * t.squeeze())

    y     = 10.0 * core #+ np.random.normal(0.0, noise_sd, size=len(core))
    return y.astype(np.float32)

def rbf_kernel(x, y, sigma=1.0):
    """
    RBF kernel matrix: K[i, j] = exp(-||x_i - y_j||^2 / (2*sigma^2))
    """
    # x shape: (N, d), y shape: (M, d)
    # cdist => pairwise Eucl. distances => shape (N, M)
    dists = torch.cdist(x, y, p=2)**2  # squared distances
    K = torch.exp(-dists / (2 * sigma**2))
    return K

def mmd_loss(z, z_prior, sigma=1.0):
    """
    Compute MMD^2 between samples z and z_prior.
    MMD^2 = E[k(z,z)] + E[k(z_prior,z_prior)] - 2 E[k(z,z_prior)]
    """
    K_zz = rbf_kernel(z, z, sigma)
    K_pp = rbf_kernel(z_prior, z_prior, sigma)
    K_zp = rbf_kernel(z, z_prior, sigma)

    mmd = K_zz.mean() + K_pp.mean() - 2 * K_zp.mean()
    return mmd



def standardize_tensor_with_scaler(X_train, X_test=None):
    # Convert the PyTorch tensor to a NumPy array
    X_train_np = X_train.numpy()

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train_standardized = scaler.fit_transform(X_train_np)

    # Convert back to a PyTorch tensor
    X_train_standardized_tensor = torch.tensor(X_train_standardized)

    if X_test is not None:
        # If test data is provided, apply the same transformation (without fitting again)
        X_test_np = X_test.numpy()
        X_test_standardized = scaler.transform(X_test_np)
        X_test_standardized_tensor = torch.tensor(X_test_standardized)

        return X_train_standardized_tensor, X_test_standardized_tensor

    return X_train_standardized_tensor
