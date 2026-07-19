from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

FEATURE_NAMES = ("S", "K", "T", "r", "sigma")


@dataclass(frozen=True)
class ModelConfig:
    input_mean: tuple[float, float, float, float, float]
    input_scale: tuple[float, float, float, float, float]
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    enforce_call_bounds: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_mean": list(self.input_mean),
            "input_scale": list(self.input_scale),
            "hidden_dims": list(self.hidden_dims),
            "enforce_call_bounds": self.enforce_call_bounds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelConfig:
        return cls(
            input_mean=tuple(float(value) for value in payload["input_mean"]),
            input_scale=tuple(float(value) for value in payload["input_scale"]),
            hidden_dims=tuple(int(value) for value in payload.get("hidden_dims", (128, 128, 64))),
            enforce_call_bounds=bool(payload.get("enforce_call_bounds", True)),
        )


class OptionMLP(nn.Module):
    """Differentiable no-dividend European-call pricer.

    Inputs remain in financial units. Scaling happens inside the computational
    graph, so autograd returns delta and vega in raw financial units without a
    manual chain-rule correction.
    """

    def __init__(
        self,
        input_mean: Iterable[float] = (100.0, 100.0, 1.05, 0.03, 0.30),
        input_scale: Iterable[float] = (30.0, 30.0, 0.55, 0.012, 0.12),
        hidden_dims: tuple[int, ...] = (128, 128, 64),
        enforce_call_bounds: bool = True,
    ) -> None:
        super().__init__()

        mean_tensor = torch.as_tensor(tuple(input_mean), dtype=torch.float32)
        scale_tensor = torch.as_tensor(tuple(input_scale), dtype=torch.float32)
        if mean_tensor.shape != (5,) or scale_tensor.shape != (5,):
            raise ValueError("input_mean and input_scale must each contain five values")
        if torch.any(scale_tensor <= 0):
            raise ValueError("all input scales must be strictly positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")

        self.register_buffer("input_mean", mean_tensor)
        self.register_buffer("input_scale", scale_tensor)
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.enforce_call_bounds = bool(enforce_call_bounds)

        layers: list[nn.Module] = []
        previous_width = 5
        for width in self.hidden_dims:
            layers.extend([nn.Linear(previous_width, width), nn.SiLU()])
            previous_width = width
        layers.append(nn.Linear(previous_width, 1))
        self.network = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def export_config(self) -> ModelConfig:
        return ModelConfig(
            input_mean=tuple(float(value) for value in self.input_mean.detach().cpu()),
            input_scale=tuple(float(value) for value in self.input_scale.detach().cpu()),
            hidden_dims=self.hidden_dims,
            enforce_call_bounds=self.enforce_call_bounds,
        )

    @classmethod
    def from_config(cls, config: ModelConfig | Mapping[str, Any]) -> OptionMLP:
        parsed = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
        return cls(
            input_mean=parsed.input_mean,
            input_scale=parsed.input_scale,
            hidden_dims=parsed.hidden_dims,
            enforce_call_bounds=parsed.enforce_call_bounds,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != 5:
            raise ValueError("inputs must have shape (batch_size, 5)")

        normalized = (inputs - self.input_mean) / self.input_scale
        raw_score = self.network(normalized).squeeze(-1)
        if not self.enforce_call_bounds:
            return raw_score

        spot = inputs[:, 0]
        strike = inputs[:, 1]
        maturity = inputs[:, 2]
        rate = inputs[:, 3]

        discounted_strike = strike * torch.exp(-rate * maturity)
        lower_bound = torch.clamp(spot - discounted_strike, min=0.0)
        upper_bound = spot
        interpolation_weight = torch.sigmoid(raw_score)
        return lower_bound + (upper_bound - lower_bound) * interpolation_weight


def price_and_greeks(
    model: nn.Module,
    inputs: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return price, delta and vega with derivatives in raw input units."""

    differentiable_inputs = inputs.clone().detach().requires_grad_(True)
    price = model(differentiable_inputs)
    gradients = torch.autograd.grad(
        outputs=price,
        inputs=differentiable_inputs,
        grad_outputs=torch.ones_like(price),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    delta = gradients[:, 0]
    vega = gradients[:, 4]
    return price, delta, vega


def save_checkpoint(
    path: str | Path,
    model: OptionMLP,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_config": model.export_config().to_dict(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output_path)
    return output_path


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[OptionMLP, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    try:
        payload = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(checkpoint_path, map_location=device)

    if not isinstance(payload, dict) or "model_config" not in payload:
        raise ValueError(
            "legacy state-dict-only checkpoint detected; retrain with "
            "python -m train.train_model"
        )

    model = OptionMLP.from_config(payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, dict(payload.get("metadata", {}))
