# train/train_model.py

import sys
import os
#  ─ Ensure project root is on PYTHONPATH ──────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from models.dml_model import OptionMLP
from losses.differential_loss import differential_loss


class OptionDataset(Dataset):
    """Dataset wrapping option pricing data."""
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.x = torch.tensor(df[['S', 'K', 'T', 'r', 'sigma']].values, dtype=torch.float32)
        self.price = torch.tensor(df['price'].values, dtype=torch.float32)
        self.delta = torch.tensor(df['delta'].values, dtype=torch.float32)
        self.vega = torch.tensor(df['vega'].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.price[idx], self.delta[idx], self.vega[idx]


def train_epoch(model, loader, optimizer, device, λ_delta, λ_vega, dataset_size):
    """Runs one training epoch, returns avg loss."""
    model.train()
    total_loss = 0.0
    for x, price, delta, vega in loader:
        x, price, delta, vega = [t.to(device) for t in (x, price, delta, vega)]
        optimizer.zero_grad()
        loss = differential_loss(model, x, price, delta, vega, λ_delta, λ_vega)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / dataset_size


# at top of train/train_model.py, after your imports:
import torch

def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_delta: float,
    lambda_vega: float,
    dataset_size: int
) -> float:
    """
    One validation epoch—returns avg composite loss.
    We explicitly ENABLE grad here and log its state to pinpoint any dropouts.
    """
    model.eval()
    total_loss = 0.0

    # Log global grad‐mode at start
    print(f"[evaluate] START | torch.is_grad_enabled(): {torch.is_grad_enabled()}")

    for batch_idx, (x, price, delta, vega) in enumerate(loader):
        # Move to device
        x, price, delta, vega = [t.to(device) for t in (x, price, delta, vega)]

        # Ensure gradients are enabled
        torch.set_grad_enabled(True)
        print(f"[evaluate][Batch {batch_idx}] After set_grad_enabled(True): torch.is_grad_enabled() = {torch.is_grad_enabled()}")

        # Now call your differential loss
        loss = differential_loss(model, x, price, delta, vega, lambda_delta, lambda_vega)
        print(f"[evaluate][Batch {batch_idx}] differential_loss returned {loss.item():.6f}")

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / dataset_size
    print(f"[evaluate] END | avg_loss = {avg_loss:.6f}\n")
    return avg_loss



def main():
    # --- Configurable parameters ---
    data_csv     = "data/option_data.csv"
    batch_size   = 256
    epochs       = 50
    lr           = 1e-3
    λ_delta      = 1.0
    λ_vega       = 1.0
    ckpt_path    = "dml_pricer_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}\n")

    # --- Step 1: Load & split data ---
    print("[Step 1] Loading data and creating train/val split...")
    df = pd.read_csv(data_csv)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv",   index=False)
    train_ds = OptionDataset("data/train.csv")
    val_ds   = OptionDataset("data/val.csv")
    train_size = len(train_ds)
    val_size   = len(val_ds)
    print(f"         → train_size: {train_size}, val_size: {val_size}\n")

    # --- Step 2: Create DataLoaders ---
    print("[Step 2] Creating DataLoaders...")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"         → Batch size: {batch_size}\n")

    # --- Step 3: Init model & optimizer ---
    print("[Step 3] Initializing model and optimizer...")
    model = OptionMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f"         → Learning rate: {lr}\n")

    # --- Step 4: Training loop ---
    print("[Step 4] Starting training...\n")
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, λ_delta, λ_vega, train_size)
        val_loss   = evaluate(model, val_loader,   device, λ_delta, λ_vega, val_size)

        print(f" Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ✅ New best model saved (val_loss={val_loss:.6f})\n")

    print("\n[Done] Training complete.")
    print(f"       Best val_loss: {best_val_loss:.6f} (checkpoint: {ckpt_path})")


if __name__ == "__main__":
    main()
