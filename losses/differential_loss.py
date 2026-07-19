from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LossScales:
    """Positive scales that make price, delta and vega losses comparable."""

    price: float
    delta: float
    vega: float

    def __post_init__(self) -> None:
        if self.price <= 0 or self.delta <= 0 or self.vega <= 0:
            raise ValueError("all loss scales must be strictly positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_targets(
        cls,
        price: Tensor,
        delta: Tensor,
        vega: Tensor,
        *,
        minimum: float = 1e-6,
    ) -> LossScales:
        if minimum <= 0:
            raise ValueError("minimum must be positive")

        def scale(values: Tensor) -> float:
            standard_deviation = float(values.detach().float().std(unbiased=False).cpu())
            return max(standard_deviation, minimum)

        return cls(
            price=scale(price),
            delta=scale(delta),
            vega=scale(vega),
        )


@dataclass(frozen=True)
class DifferentialLossWeights:
    price: float = 1.0
    delta: float = 1.0
    vega: float = 1.0

    def __post_init__(self) -> None:
        if self.price < 0 or self.delta < 0 or self.vega < 0:
            raise ValueError("loss weights cannot be negative")
        if self.price + self.delta + self.vega <= 0:
            raise ValueError("at least one loss weight must be positive")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def differential_loss(
    model: nn.Module,
    inputs: Tensor,
    true_price: Tensor,
    true_delta: Tensor,
    true_vega: Tensor,
    *,
    scales: LossScales,
    weights: DifferentialLossWeights | None = None,
    create_graph: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Composite loss on price and raw-unit delta/vega labels."""

    weights = weights or DifferentialLossWeights()
    differentiable_inputs = inputs.clone().detach().requires_grad_(True)
    predicted_price = model(differentiable_inputs)
    gradients = torch.autograd.grad(
        outputs=predicted_price,
        inputs=differentiable_inputs,
        grad_outputs=torch.ones_like(predicted_price),
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    predicted_delta = gradients[:, 0]
    predicted_vega = gradients[:, 4]

    price_loss = F.mse_loss(
        (predicted_price - true_price) / scales.price,
        torch.zeros_like(true_price),
    )
    delta_loss = F.mse_loss(
        (predicted_delta - true_delta) / scales.delta,
        torch.zeros_like(true_delta),
    )
    vega_loss = F.mse_loss(
        (predicted_vega - true_vega) / scales.vega,
        torch.zeros_like(true_vega),
    )

    total_loss = (
        weights.price * price_loss
        + weights.delta * delta_loss
        + weights.vega * vega_loss
    )
    metrics = {
        "total": total_loss.detach(),
        "price_normalized_mse": price_loss.detach(),
        "delta_normalized_mse": delta_loss.detach(),
        "vega_normalized_mse": vega_loss.detach(),
        "predicted_price": predicted_price.detach(),
        "predicted_delta": predicted_delta.detach(),
        "predicted_vega": predicted_vega.detach(),
    }
    return total_loss, metrics


def price_only_loss(
    model: nn.Module,
    inputs: Tensor,
    true_price: Tensor,
    *,
    price_scale: float,
) -> Tensor:
    if price_scale <= 0:
        raise ValueError("price_scale must be positive")
    prediction = model(inputs)
    return F.mse_loss(
        (prediction - true_price) / price_scale,
        torch.zeros_like(true_price),
    )


def loss_metadata(
    scales: LossScales,
    weights: DifferentialLossWeights,
) -> dict[str, Any]:
    return {
        "scales": scales.to_dict(),
        "weights": weights.to_dict(),
        "greek_units": {
            "delta": "price change per one unit change in spot",
            "vega": "price change per 1.00 change in volatility",
        },
    }
