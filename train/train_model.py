# train/train_model.py

import sys
import os
import itertools
# ─ Ensure project root is on PYTHONPATH ──────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from models.dml_model import OptionMLP
from losses.differential_loss import differential_loss


class OptionDataset(Dataset):
    """Wraps CSV of [S, K, T, r, sigma, price, delta, vega] into torch tensors."""
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.x     = torch.tensor(df[['S','K','T','r','sigma']].values, dtype=torch.float32)
        self.price = torch.tensor(df['price'].values,                         dtype=torch.float32)
        self.delta = torch.tensor(df['delta'].values,                         dtype=torch.float32)
        self.vega  = torch.tensor(df['vega'].values,                          dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.price[idx], self.delta[idx], self.vega[idx]


def train_epoch(model, loader, optimizer, device, λ_delta, λ_vega, N):
    """Runs one training epoch—returns dict of total-loss and RMSE metrics."""
    model.train()
    sums = {'total':0.0, 'price':0.0, 'delta':0.0, 'vega':0.0}

    for x, price, delta, vega in loader:
        x, price, delta, vega = [t.to(device) for t in (x, price, delta, vega)]
        optimizer.zero_grad()
        total_loss, mets = differential_loss(model, x, price, delta, vega, λ_delta, λ_vega)
        total_loss.backward()
        optimizer.step()

        b = x.size(0)
        sums['total'] += total_loss.item() * b
        sums['price'] += mets['price'].item() * b
        sums['delta'] += mets['delta'].item() * b
        sums['vega']  += mets['vega'].item()  * b

    return {
        'total': sums['total'] / N,
        'price': (sums['price'] / N)**0.5,
        'delta': (sums['delta'] / N)**0.5,
        'vega':  (sums['vega']  / N)**0.5
    }


def evaluate(model, loader, device, λ_delta, λ_vega, N):
    """One validation epoch—returns dict of total-loss and RMSE metrics."""
    model.eval()
    sums = {'total':0.0, 'price':0.0, 'delta':0.0, 'vega':0.0}

    for x, price, delta, vega in loader:
        x, price, delta, vega = [t.to(device) for t in (x, price, delta, vega)]
        torch.set_grad_enabled(True)  # ensure AAD works here
        total_loss, mets = differential_loss(model, x, price, delta, vega, λ_delta, λ_vega)

        b = x.size(0)
        sums['total'] += total_loss.item() * b
        sums['price'] += mets['price'].item() * b
        sums['delta'] += mets['delta'].item() * b
        sums['vega']  += mets['vega'].item()  * b

    return {
        'total': sums['total'] / N,
        'price': (sums['price'] / N)**0.5,
        'delta': (sums['delta'] / N)**0.5,
        'vega':  (sums['vega']  / N)**0.5
    }


def main():
    # ─── Hyperparameter grids & settings ─────────────────────────────
    lr_list      = [1e-3, 5e-4]
    bs_list      = [128, 256]
    λ_delta_list = [0.5, 1.0, 2.0]
    λ_vega_list  = [0.5, 1.0, 2.0]
    epochs       = 20  # shorter for sweep

    # ─── Tracking variables (pre-initialized) ────────────────────────
    best_score = float("inf")
    # Initialize best_cfg as a tuple so indexing is always valid
    best_cfg   = (lr_list[0], bs_list[0], λ_delta_list[0], λ_vega_list[0])
    # Predefine m_val so Pylance knows it's always bound
    m_val      = {'total': float("inf"), 'price': 0.0, 'delta': 0.0, 'vega': 0.0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}\n")

    # Load data once
    df = pd.read_csv("data/option_data.csv")

    # ─── Hyperparameter sweep ─────────────────────────────────────────
    for lr, batch_size, λ_delta, λ_vega in itertools.product(
        lr_list, bs_list, λ_delta_list, λ_vega_list
    ):
        print(f"→ Trying lr={lr}, bs={batch_size}, λΔ={λ_delta}, λν={λ_vega}")
        # Split & save
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        os.makedirs("data", exist_ok=True)
        train_df.to_csv("data/train.csv", index=False)
        val_df.to_csv("data/val.csv",   index=False)

        # Create datasets & loaders
        train_ds = OptionDataset("data/train.csv")
        val_ds   = OptionDataset("data/val.csv")
        N_train, N_val = len(train_ds), len(val_ds)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

        # Init model & optimizer
        model = OptionMLP().to(device)
        opt   = optim.Adam(model.parameters(), lr=lr)

        # Quick train/validate
        for epoch in range(1, epochs + 1):
            m_tr = train_epoch(model, train_loader, opt, device, λ_delta, λ_vega, N_train)
            m_val= evaluate(   model, val_loader,   device, λ_delta, λ_vega, N_val)

            if epoch % 5 == 0:
                print(
                    f"  epoch {epoch:02d} | total-trn:{m_tr['total']:.3f} val:{m_val['total']:.3f} "
                    f"| price-RMSE:{m_tr['price']:.2f}/{m_val['price']:.2f} "
                    f"| delta-RMSE:{m_tr['delta']:.2f}/{m_val['delta']:.2f} "
                    f"| vega-RMSE:{m_tr['vega']:.2f}/{m_val['vega']:.2f}"
                )

        # Update best config
        if m_val['total'] < best_score:
            best_score = m_val['total']
            best_cfg   = (lr, batch_size, λ_delta, λ_vega)
            torch.save(model.state_dict(), "dml_pricer_best.pth")
            print(f"  ✅ New best config saved: total-val={best_score:.4f}\n")

    # ─── Final results ────────────────────────────────────────────────
    print(
        f"\n🏆 Best config: lr={best_cfg[0]}, bs={best_cfg[1]}, "
        f"λΔ={best_cfg[2]}, λν={best_cfg[3]} → val-total={best_score:.4f}"
    )
    print("Model checkpoint saved to dml_pricer_best.pth")


if __name__ == "__main__":
    main()
