from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from data.bs_data_generator import SamplingBounds, generate_synthetic_data
from losses.differential_loss import (
    DifferentialLossWeights,
    LossScales,
    differential_loss,
    loss_metadata,
    price_only_loss,
)
from models.dml_model import OptionMLP, price_and_greeks, save_checkpoint

FEATURE_COLUMNS = ["S", "K", "T", "r", "sigma"]
TARGET_COLUMNS = ["price", "delta", "vega"]


@dataclass(frozen=True)
class TrainingConfig:
    samples: int = 20_000
    seed: int = 42
    epochs: int = 80
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 12
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    gradient_clip: float = 5.0

    def validate(self) -> None:
        if self.samples < 100:
            raise ValueError("samples must be at least 100")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must sum to less than one")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on runner
        torch.cuda.manual_seed_all(seed)


def split_frame(
    frame: pd.DataFrame,
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(frame) < 3:
        raise ValueError("frame must contain at least three rows")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    test_count = max(1, int(round(len(frame) * test_fraction)))
    validation_count = max(1, int(round(len(frame) * validation_fraction)))
    if test_count + validation_count >= len(frame):
        raise ValueError("split fractions leave no training rows")

    test_indices = indices[:test_count]
    validation_indices = indices[test_count : test_count + validation_count]
    training_indices = indices[test_count + validation_count :]
    return (
        frame.iloc[training_indices].reset_index(drop=True),
        frame.iloc[validation_indices].reset_index(drop=True),
        frame.iloc[test_indices].reset_index(drop=True),
    )


def frame_to_tensors(frame: pd.DataFrame) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    missing = set(FEATURE_COLUMNS + TARGET_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing columns: {sorted(missing)}")

    features = torch.tensor(frame[FEATURE_COLUMNS].to_numpy(), dtype=torch.float32)
    price = torch.tensor(frame["price"].to_numpy(), dtype=torch.float32)
    delta = torch.tensor(frame["delta"].to_numpy(), dtype=torch.float32)
    vega = torch.tensor(frame["vega"].to_numpy(), dtype=torch.float32)
    return features, price, delta, vega


def make_loader(
    tensors: tuple[Tensor, Tensor, Tensor, Tensor],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )


def build_model(
    training_features: Tensor,
    hidden_dims: tuple[int, ...],
) -> OptionMLP:
    mean = training_features.mean(dim=0)
    scale = training_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return OptionMLP(
        input_mean=mean.tolist(),
        input_scale=scale.tolist(),
        hidden_dims=hidden_dims,
        enforce_call_bounds=True,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    price_predictions: list[Tensor] = []
    delta_predictions: list[Tensor] = []
    vega_predictions: list[Tensor] = []
    price_targets: list[Tensor] = []
    delta_targets: list[Tensor] = []
    vega_targets: list[Tensor] = []

    for inputs, price, delta, vega in loader:
        inputs = inputs.to(device)
        predicted_price, predicted_delta, predicted_vega = price_and_greeks(
            model,
            inputs,
            create_graph=False,
        )
        price_predictions.append(predicted_price.detach().cpu())
        delta_predictions.append(predicted_delta.detach().cpu())
        vega_predictions.append(predicted_vega.detach().cpu())
        price_targets.append(price)
        delta_targets.append(delta)
        vega_targets.append(vega)

    predicted_price = torch.cat(price_predictions)
    predicted_delta = torch.cat(delta_predictions)
    predicted_vega = torch.cat(vega_predictions)
    true_price = torch.cat(price_targets)
    true_delta = torch.cat(delta_targets)
    true_vega = torch.cat(vega_targets)

    def rmse(prediction: Tensor, target: Tensor) -> float:
        return float(torch.sqrt(torch.mean((prediction - target) ** 2)))

    def mae(prediction: Tensor, target: Tensor) -> float:
        return float(torch.mean(torch.abs(prediction - target)))

    price_rmse = rmse(predicted_price, true_price)
    mean_abs_price = max(float(torch.mean(torch.abs(true_price))), 1e-8)
    return {
        "price_rmse": price_rmse,
        "price_mae": mae(predicted_price, true_price),
        "price_nrmse_pct": 100.0 * price_rmse / mean_abs_price,
        "delta_rmse": rmse(predicted_delta, true_delta),
        "delta_mae": mae(predicted_delta, true_delta),
        "vega_rmse": rmse(predicted_vega, true_vega),
        "vega_mae": mae(predicted_vega, true_vega),
        "test_rows": float(len(true_price)),
    }


def train_candidate(
    model: OptionMLP,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    scales: LossScales,
    weights: DifferentialLossWeights,
    config: TrainingConfig,
    device: torch.device,
    differential: bool,
) -> tuple[OptionMLP, list[dict[str, float]]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_validation_score = math.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        rows_seen = 0

        for inputs, price, delta, vega in training_loader:
            inputs = inputs.to(device)
            price = price.to(device)
            delta = delta.to(device)
            vega = vega.to(device)

            optimizer.zero_grad(set_to_none=True)
            if differential:
                loss, _ = differential_loss(
                    model,
                    inputs,
                    price,
                    delta,
                    vega,
                    scales=scales,
                    weights=weights,
                    create_graph=True,
                )
            else:
                loss = price_only_loss(
                    model,
                    inputs,
                    price,
                    price_scale=scales.price,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            batch_size = len(inputs)
            running_loss += float(loss.detach().cpu()) * batch_size
            rows_seen += batch_size

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device=device,
        )
        if differential:
            validation_score = (
                validation_metrics["price_rmse"] / scales.price
                + validation_metrics["delta_rmse"] / scales.delta
                + validation_metrics["vega_rmse"] / scales.vega
            )
        else:
            validation_score = validation_metrics["price_rmse"] / scales.price
        epoch_record = {
            "epoch": float(epoch),
            "training_loss": running_loss / max(rows_seen, 1),
            "validation_score": validation_score,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_record)

        if validation_score < best_validation_score - 1e-6:
            best_validation_score = validation_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            label = "DML" if differential else "price-only"
            print(
                f"[{label}] epoch={epoch:03d} "
                f"train_loss={epoch_record['training_loss']:.6f} "
                f"val_score={validation_score:.6f}"
            )

        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, history


def train_pipeline(
    config: TrainingConfig,
    *,
    compare_baseline: bool = False,
    output_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    config.validate()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = SamplingBounds()
    frame = generate_synthetic_data(
        n_samples=config.samples,
        seed=config.seed,
        bounds=bounds,
        augment=False,
    )
    train_frame, validation_frame, test_frame = split_frame(
        frame,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )

    train_tensors = frame_to_tensors(train_frame)
    validation_tensors = frame_to_tensors(validation_frame)
    test_tensors = frame_to_tensors(test_frame)
    scales = LossScales.from_targets(
        train_tensors[1],
        train_tensors[2],
        train_tensors[3],
    )
    weights = DifferentialLossWeights(price=1.0, delta=1.0, vega=1.0)

    train_loader = make_loader(
        train_tensors,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    validation_loader = make_loader(
        validation_tensors,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + 1,
    )
    test_loader = make_loader(
        test_tensors,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + 2,
    )

    model = build_model(train_tensors[0], config.hidden_dims)
    model, dml_history = train_candidate(
        model,
        train_loader,
        validation_loader,
        scales=scales,
        weights=weights,
        config=config,
        device=device,
        differential=True,
    )
    dml_metrics = evaluate_model(model, test_loader, device=device)

    baseline_metrics: dict[str, float] | None = None
    if compare_baseline:
        # Reset both the global model-initialisation seed and the DataLoader
        # generators. This gives the price-only benchmark the same starting
        # weights, split and epoch-by-epoch sample order as the DML candidate.
        set_seed(config.seed)
        baseline_train_loader = make_loader(
            train_tensors,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed,
        )
        baseline_validation_loader = make_loader(
            validation_tensors,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed + 1,
        )
        baseline_test_loader = make_loader(
            test_tensors,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed + 2,
        )
        baseline = build_model(train_tensors[0], config.hidden_dims)
        baseline, _ = train_candidate(
            baseline,
            baseline_train_loader,
            baseline_validation_loader,
            scales=scales,
            weights=weights,
            config=config,
            device=device,
            differential=False,
        )
        baseline_metrics = evaluate_model(
            baseline,
            baseline_test_loader,
            device=device,
        )

    metadata: dict[str, Any] = {
        "training_config": asdict(config),
        "sampling_bounds": bounds.to_dict(),
        "loss": loss_metadata(scales, weights),
        "metrics": dml_metrics,
        "baseline_metrics": baseline_metrics,
        "device_used_for_training": str(device),
        "training_history": dml_history,
        "model_scope": "no-dividend European calls under Black-Scholes",
    }

    output_directory = Path(output_dir)
    checkpoint_path = save_checkpoint(
        output_directory / "dml_option_pricer.pt",
        model,
        metadata=metadata,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics_path = output_directory / "metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    benchmark_path = output_directory / "benchmark_summary.md"
    benchmark_path.write_text(
        render_benchmark_summary(metadata),
        encoding="utf-8",
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved benchmark summary to {benchmark_path}")
    print(json.dumps(dml_metrics, indent=2))
    return metadata


def _improvement_pct(candidate: float, baseline: float) -> float | None:
    """Return percentage error reduction versus baseline when well-defined."""
    if not math.isfinite(candidate) or not math.isfinite(baseline) or baseline <= 0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def render_benchmark_summary(metadata: dict[str, Any]) -> str:
    """Render a compact Markdown benchmark directly from training metadata."""
    metrics = metadata.get("metrics") or {}
    baseline = metadata.get("baseline_metrics")
    config = metadata.get("training_config") or {}

    lines = [
        "# Generated Benchmark Summary",
        "",
        "This file is generated by `python -m train.train_model` from the same "
        "metadata stored in `artifacts/metrics.json`.",
        "",
        f"- Scope: {metadata.get('model_scope', 'not recorded')}",
        f"- Seed: {config.get('seed', 'not recorded')}",
        f"- Training samples: {config.get('samples', 'not recorded')}",
        f"- Held-out test rows: {int(metrics.get('test_rows', 0))}",
        "",
    ]

    if not baseline:
        lines.extend(
            [
                "A price-only benchmark was not run for this checkpoint.",
                "",
                "Re-run with `--compare-baseline` to produce a controlled comparison.",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "The DML and price-only models use the same split, architecture, "
            "initialisation seed and shuffled batch order.",
            "",
            "| Metric | DML | Price-only baseline | Error reduction |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("Price RMSE", "price_rmse"),
        ("Delta RMSE", "delta_rmse"),
        ("Vega RMSE", "vega_rmse"),
    ):
        candidate_value = float(metrics[key])
        baseline_value = float(baseline[key])
        improvement = _improvement_pct(candidate_value, baseline_value)
        improvement_text = "n/a" if improvement is None else f"{improvement:.1f}%"
        lines.append(
            f"| {label} | {candidate_value:.6f} | "
            f"{baseline_value:.6f} | {improvement_text} |"
        )

    lines.extend(
        [
            "",
            "These results measure approximation error on a held-out synthetic "
            "Black-Scholes test set; they are not evidence of market calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the differential option pricer")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - exercised through CLI
    args = parse_args()
    config = TrainingConfig(
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        patience=args.patience,
    )
    train_pipeline(
        config,
        compare_baseline=args.compare_baseline,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
