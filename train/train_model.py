# train/train_model.py

import sys
import os
import itertools

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dml_model import OptionMLP
from losses.differential_loss import differential_loss
from data.bs_data_generator import split_base_data, augment_dataframe


FEATURE_COLS = ["S", "K", "T", "r", "sigma"]
TARGET_COLS = ["price", "delta", "vega"]


class OptionDataset(Dataset):
    """Wrap a DataFrame into normalized tensors."""

    def __init__(self, df: pd.DataFrame, feature_mean: pd.Series, feature_std: pd.Series):
        x = df[FEATURE_COLS].copy()
        x = (x - feature_mean) / feature_std

        self.x = torch.tensor(x.values, dtype=torch.float32)
        self.price = torch.tensor(df["price"].values, dtype=torch.float32)
        self.delta = torch.tensor(df["delta"].values, dtype=torch.float32)
        self.vega = torch.tensor(df["vega"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.price[idx], self.delta[idx], self.vega[idx]


def train_epoch(model, loader, optimizer, device, lambda_delta, lambda_vega, n_obs, s_scale, sigma_scale):
    model.train()
    sums = {"total": 0.0, "price": 0.0, "delta": 0.0, "vega": 0.0}

    for x, price, delta, vega in loader:
        x, price, delta, vega = [t.to(device) for t in (x, price, delta, vega)]
        optimizer.zero_grad()
        total_loss, mets = differential_loss(
            model,
            x,
            price,
            delta,
            vega,
            lambda_delta,
            lambda_vega,
            s_scale=s_scale,
            sigma_scale=sigma_scale,
        )
        total_loss.backward()
        optimizer.step()

        b = x.size(0)
        sums["total"] += total_loss.item() * b
        sums["price"] += mets["price"].item() * b
        sums["delta"] += mets["delta"].item() * b
        sums["vega"] += mets["vega"].item() * b

    return {
        "total": sums["total"] / n_obs,
        "price": (sums["price"] / n_obs) ** 0.5,
        "delta": (sums["delta"] / n_obs) ** 0.5,
        "vega": (sums["vega"] / n_obs) ** 0.5,
    }


def evaluate(model, loader, device, lambda_delta, lambda_vega, n_obs, s_scale, sigma_scale):
    model.eval()
    sums = {"total": 0.0, "price": 0.0, "delta": 0.0, "vega": 0.0}

    for x, price, delta, vega in loader:
        x, price, delta, vega = [t.to(device) for t in (x, price, delta, vega)]
        torch.set_grad_enabled(True)
        total_loss, mets = differential_loss(
            model,
            x,
            price,
            delta,
            vega,
            lambda_delta,
            lambda_vega,
            s_scale=s_scale,
            sigma_scale=sigma_scale,
        )

        b = x.size(0)
        sums["total"] += total_loss.item() * b
        sums["price"] += mets["price"].item() * b
        sums["delta"] += mets["delta"].item() * b
        sums["vega"] += mets["vega"].item() * b

    return {
        "total": sums["total"] / n_obs,
        "price": (sums["price"] / n_obs) ** 0.5,
        "delta": (sums["delta"] / n_obs) ** 0.5,
        "vega": (sums["vega"] / n_obs) ** 0.5,
    }


def compute_test_metrics(model, test_df, feature_mean, feature_std, device):
    model.eval()
    x_df = (test_df[FEATURE_COLS] - feature_mean) / feature_std
    x = torch.tensor(x_df.values, dtype=torch.float32, device=device).requires_grad_(True)
    y_true = test_df["price"].to_numpy()
    delta_true = test_df["delta"].to_numpy()
    vega_true = test_df["vega"].to_numpy()

    pred_price = model(x)
    grads = torch.autograd.grad(
        outputs=pred_price,
        inputs=x,
        grad_outputs=torch.ones_like(pred_price),
        create_graph=False,
    )[0]

    pred_price_np = pred_price.detach().cpu().numpy()
    pred_delta_np = grads[:, 0].detach().cpu().numpy() / float(feature_std["S"])
    pred_vega_np = grads[:, 4].detach().cpu().numpy() / float(feature_std["sigma"])

    price_rmse = float(np.sqrt(np.mean((pred_price_np - y_true) ** 2)))
    delta_rmse = float(np.sqrt(np.mean((pred_delta_np - delta_true) ** 2)))
    vega_rmse = float(np.sqrt(np.mean((pred_vega_np - vega_true) ** 2)))

    norm_denom = float(np.mean(np.abs(y_true))) if np.mean(np.abs(y_true)) > 1e-8 else 1.0
    nrmse = price_rmse / norm_denom

    mape_mask = np.abs(y_true) > 1e-3
    if mape_mask.any():
        mape = float(np.mean(np.abs((pred_price_np[mape_mask] - y_true[mape_mask]) / y_true[mape_mask])) * 100.0)
    else:
        mape = float("nan")

    abs_err = np.abs(pred_price_np - y_true)
    cutoff = np.quantile(abs_err, 0.9)
    worst_decile_error = float(abs_err[abs_err >= cutoff].mean())

    return {
        "price_rmse": price_rmse,
        "nrmse": nrmse,
        "mape_ex_near_zero_pct": mape,
        "delta_rmse": delta_rmse,
        "vega_rmse": vega_rmse,
        "worst_decile_abs_error": worst_decile_error,
    }


def main():
    lr_list = [1e-3, 5e-4]
    bs_list = [128, 256]
    lambda_grid = [0.5, 1.0, 2.0]
    epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}\n")

    os.makedirs("data", exist_ok=True)

    # Generate and split base samples before any augmentation.
    train_base_df, val_df, test_df = split_base_data(n_samples=20000, seed=42, val_size=0.15, test_size=0.15)

    # Augment training data only.
    train_aug_df = augment_dataframe(train_base_df, seed=43, noise_std=0.01)
    train_df = pd.concat([train_base_df, train_aug_df], ignore_index=True)

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    feature_mean = train_df[FEATURE_COLS].mean()
    feature_std = train_df[FEATURE_COLS].std().replace(0.0, 1.0)

    train_ds = OptionDataset(train_df, feature_mean, feature_std)
    val_ds = OptionDataset(val_df, feature_mean, feature_std)

    n_train, n_val = len(train_ds), len(val_ds)

    # Variance-aware base weights.
    price_var = float(train_df["price"].var())
    delta_var = float(train_df["delta"].var())
    vega_var = float(train_df["vega"].var())
    base_lambda_delta = price_var / max(delta_var, 1e-8)
    base_lambda_vega = price_var / max(vega_var, 1e-8)

    best_score = float("inf")
    best_cfg = (lr_list[0], bs_list[0], lambda_grid[0], lambda_grid[0])

    for lr, batch_size, scale_delta, scale_vega in itertools.product(
        lr_list, bs_list, lambda_grid, lambda_grid
    ):
        lambda_delta = base_lambda_delta * scale_delta
        lambda_vega = base_lambda_vega * scale_vega
        print(
            f"Trying lr={lr}, bs={batch_size}, lambda_delta={lambda_delta:.4f} "
            f"(x{scale_delta}), lambda_vega={lambda_vega:.4f} (x{scale_vega})"
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = OptionMLP().to(device)
        opt = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(1, epochs + 1):
            m_tr = train_epoch(
                model,
                train_loader,
                opt,
                device,
                lambda_delta,
                lambda_vega,
                n_train,
                float(feature_std["S"]),
                float(feature_std["sigma"]),
            )
            m_val = evaluate(
                model,
                val_loader,
                device,
                lambda_delta,
                lambda_vega,
                n_val,
                float(feature_std["S"]),
                float(feature_std["sigma"]),
            )

            if epoch % 5 == 0:
                print(
                    f"  epoch {epoch:02d} | total-trn:{m_tr['total']:.4f} val:{m_val['total']:.4f} "
                    f"| price-RMSE:{m_tr['price']:.4f}/{m_val['price']:.4f} "
                    f"| delta-RMSE:{m_tr['delta']:.4f}/{m_val['delta']:.4f} "
                    f"| vega-RMSE:{m_tr['vega']:.4f}/{m_val['vega']:.4f}"
                )

        if m_val["total"] < best_score:
            best_score = m_val["total"]
            best_cfg = (lr, batch_size, scale_delta, scale_vega)
            torch.save(model.state_dict(), "dml_pricer_best.pth")
            print(f"  New best config saved: total-val={best_score:.6f}\n")

    # Final untouched test evaluation once.
    best_model = OptionMLP().to(device)
    best_model.load_state_dict(torch.load("dml_pricer_best.pth", map_location=device))
    test_metrics = compute_test_metrics(best_model, test_df, feature_mean, feature_std, device)

    print(
        f"\nBest config: lr={best_cfg[0]}, bs={best_cfg[1]}, "
        f"lambda_delta-scale={best_cfg[2]}, lambda_vega-scale={best_cfg[3]} -> val-total={best_score:.6f}"
    )
    print("\nUntouched test-set metrics")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
