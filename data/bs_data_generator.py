# data/bs_data_generator.py

import numpy as np
import pandas as pd
from scipy.stats import norm


def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def black_scholes_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def generate_synthetic_data(n_samples=10000, 
                            seed=42, 
                            augment: bool = True, 
                            noise_std: float = 0.01) -> pd.DataFrame:
    """
    Generates BSM prices + Greeks, plus optional percent‐noise augmentation.
    """
    np.random.seed(seed)
    S     = np.random.uniform(50, 150, n_samples)
    K     = np.random.uniform(50, 150, n_samples)
    T     = np.random.uniform(0.1, 2.0, n_samples)
    r     = np.random.uniform(0.01, 0.05, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)

    # base dataset
    df = pd.DataFrame({
        'S':      S,
        'K':      K,
        'T':      T,
        'r':      r,
        'sigma':  sigma,
        'price':  black_scholes_call_price(S, K, T, r, sigma),
        'delta':  black_scholes_delta(S, K, T, r, sigma),
        'vega':   black_scholes_vega(S, K, T, r, sigma)
    })

    if augment:
        # percent-noise augmentation
        X = np.stack([S, K, T, r, sigma], axis=1)
        noise = np.random.normal(0, noise_std, size=X.shape)
        X_aug = X * (1 + noise)
        S2, K2, T2, r2, sigma2 = X_aug.T

        df_aug = pd.DataFrame({
            'S':      S2,
            'K':      K2,
            'T':      T2,
            'r':      r2,
            'sigma':  sigma2,
            'price':  black_scholes_call_price(S2, K2, T2, r2, sigma2),
            'delta':  black_scholes_delta(S2, K2, T2, r2, sigma2),
            'vega':   black_scholes_vega(S2, K2, T2, r2, sigma2)
        })
        df = pd.concat([df, df_aug], ignore_index=True)

    return df


def augment_dataframe(df: pd.DataFrame, seed: int = 42, noise_std: float = 0.01) -> pd.DataFrame:
    """Create near-neighbor perturbations for training data only."""
    rng = np.random.default_rng(seed)
    X = df[['S', 'K', 'T', 'r', 'sigma']].to_numpy(copy=True)
    noise = rng.normal(0.0, noise_std, size=X.shape)
    X_aug = X * (1.0 + noise)
    S2, K2, T2, r2, sigma2 = X_aug.T

    df_aug = pd.DataFrame({
        'S': S2,
        'K': K2,
        'T': T2,
        'r': r2,
        'sigma': sigma2,
        'price': black_scholes_call_price(S2, K2, T2, r2, sigma2),
        'delta': black_scholes_delta(S2, K2, T2, r2, sigma2),
        'vega': black_scholes_vega(S2, K2, T2, r2, sigma2),
    })
    return df_aug


def split_base_data(
    n_samples: int = 10000,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate base samples and split into train/val/test before augmentation."""
    from sklearn.model_selection import train_test_split

    base_df = generate_synthetic_data(n_samples=n_samples, seed=seed, augment=False)
    train_df, temp_df = train_test_split(base_df, test_size=(val_size + test_size), random_state=seed)
    rel_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test_size, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=20000, augment=True)
    df.to_csv("data/option_data.csv", index=False)
    print("✅ Generated and saved augmented data to data/option_data.csv")
