import numpy as np
from scipy.stats import beta as beta_dist


def generate_tcga_outcomes(x_df, t_vec, seed=42):
    """
    Generate outcomes for a batch of samples using the TCGA outcome function.

    Parameters:
    - x_df: pandas DataFrame of shape (n_samples, n_features)
    - t_vec: numpy array or list of treatment values, shape (n_samples,)
    - seed: optional int, for reproducibility

    Returns:
    - y: numpy array of shape (n_samples,)
    - v1, v2, v3: weight vectors used
    """
    if seed is not None:
        np.random.seed(seed)

    X = x_df.to_numpy()
    t_vec = np.asarray(t_vec).flatten()
    
    if X.shape[0] != t_vec.shape[0]:
        raise ValueError("Mismatch: x has %d rows but t has %d values" % (X.shape[0], t_vec.shape[0]))

    d = X.shape[1]

    def sample_and_normalize():
        v = np.random.randn(d)
        return v / np.linalg.norm(v)

    v1 = sample_and_normalize()
    v2 = sample_and_normalize()
    v3 = sample_and_normalize()

    # Dot products per sample
    dot1 = X @ v1         # shape (n_samples,)
    dot2 = X @ v2         # shape (n_samples,)
    dot3 = X @ v3         # shape (n_samples,)

    y = 10 * (dot1 + 12 * dot2 * t_vec - 12 * dot3 * t_vec**2)
    return y, v1, v2, v3


def assign_treatments_tcga(df, alpha=2.0, seed=42):
    """
    Assigns treatment values to each row in a DataFrame using the Bica et al. (2020) TCGA rule:
        t ~ Beta(alpha, beta(x))
    where beta(x) = 2*(alpha - 1)*(v2^T x)/(v3^T x) + 2 - alpha

    Parameters:
    - df: pandas DataFrame of shape (n_samples, n_covariates), numerical covariates
    - alpha: scalar, treatment selection bias
    - seed: optional int for reproducibility

    Returns:
    - treatments: numpy array of shape (n_samples,)
    - v2, v3: vectors used in the beta(x) computation
    """
    if seed is not None:
        np.random.seed(seed)

    X = df.to_numpy()
    d = X.shape[1]

    def sample_and_normalize():
        v = np.random.randn(d)
        return v / np.linalg.norm(v)

    v2 = sample_and_normalize()
    v3 = sample_and_normalize()

    dot_v2_X = X @ v2   # shape (n_samples,)
    dot_v3_X = X @ v3   # shape (n_samples,)
    
    # Avoid division by zero
    dot_v3_X = np.where(np.isclose(dot_v3_X, 0.0), 1e-8, dot_v3_X)

    beta_vals = (2 * (alpha - 1) * dot_v2_X / dot_v3_X) + 2 - alpha
    beta_vals = np.clip(beta_vals, 1e-3, 100)  # ensure positivity and numerical stability

    treatments = beta_dist.rvs(alpha, beta_vals)
    return treatments, v2, v3


def generate_news_outcomes(x_df, t_vec, seed=42):
    """
    Generate outcomes for a batch of samples using the NEWS outcome function:
        y = 10 * (v1^T x + sin((v2^T x / v3^T x) * π * t))

    Parameters:
    - x_df: pandas DataFrame of shape (n_samples, n_features)
    - t_vec: numpy array or list of treatment values, shape (n_samples,)
    - seed: optional int, for reproducibility

    Returns:
    - y: numpy array of shape (n_samples,)
    - v1, v2, v3: weight vectors used
    """
    if seed is not None:
        np.random.seed(seed)

    X = x_df.to_numpy()
    t_vec = np.asarray(t_vec).flatten()

    if X.shape[0] != t_vec.shape[0]:
        raise ValueError(f"Mismatch: x has {X.shape[0]} rows but t has {t_vec.shape[0]} values")

    d = X.shape[1]

    def sample_and_normalize():
        v = np.random.randn(d)
        return v / np.linalg.norm(v)

    v1 = sample_and_normalize()
    v2 = sample_and_normalize()
    v3 = sample_and_normalize()

    dot1 = X @ v1       # v1^T x
    dot2 = X @ v2       # v2^T x
    dot3 = X @ v3       # v3^T x

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(dot3 != 0, dot2 / dot3, 0.0)  # avoid division by zero

    y = 10 * (dot1 + np.sin(ratio * np.pi * t_vec))
    return y, v1, v2, v3


