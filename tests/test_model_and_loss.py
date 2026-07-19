from pathlib import Path

import pytest
import torch
from torch import nn

from data.bs_data_generator import (
    black_scholes_call_price,
    black_scholes_delta,
    black_scholes_vega,
)
from losses.differential_loss import (
    DifferentialLossWeights,
    LossScales,
    differential_loss,
)
from models.dml_model import (
    OptionMLP,
    load_checkpoint,
    price_and_greeks,
    save_checkpoint,
)


class TorchBlackScholesCall(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        spot, strike, maturity, rate, volatility = inputs.unbind(dim=1)
        root_t = torch.sqrt(maturity)
        d1 = (
            torch.log(spot / strike)
            + (rate + 0.5 * volatility**2) * maturity
        ) / (volatility * root_t)
        d2 = d1 - volatility * root_t
        return spot * torch.special.ndtr(d1) - strike * torch.exp(
            -rate * maturity
        ) * torch.special.ndtr(d2)


def test_model_enforces_call_price_bounds_and_has_raw_unit_greeks():
    model = OptionMLP()
    inputs = torch.tensor(
        [[80.0, 100.0, 0.5, 0.03, 0.20], [120.0, 100.0, 1.5, 0.04, 0.30]],
        dtype=torch.float32,
    )
    price, delta, vega = price_and_greeks(model, inputs)
    lower = torch.clamp(
        inputs[:, 0] - inputs[:, 1] * torch.exp(-inputs[:, 3] * inputs[:, 2]),
        min=0.0,
    )

    assert torch.all(price >= lower)
    assert torch.all(price <= inputs[:, 0])
    assert torch.isfinite(delta).all()
    assert torch.isfinite(vega).all()


def test_differential_loss_is_near_zero_for_analytic_model():
    inputs = torch.tensor(
        [[80.0, 100.0, 0.5, 0.03, 0.20], [100.0, 100.0, 1.0, 0.05, 0.20]],
        dtype=torch.float64,
    )
    numpy_inputs = inputs.numpy()
    price = torch.tensor(
        black_scholes_call_price(*numpy_inputs.T),
        dtype=torch.float64,
    )
    delta = torch.tensor(
        black_scholes_delta(*numpy_inputs.T),
        dtype=torch.float64,
    )
    vega = torch.tensor(
        black_scholes_vega(*numpy_inputs.T),
        dtype=torch.float64,
    )
    scales = LossScales(price=10.0, delta=1.0, vega=40.0)

    loss, metrics = differential_loss(
        TorchBlackScholesCall(),
        inputs,
        price,
        delta,
        vega,
        scales=scales,
        weights=DifferentialLossWeights(),
        create_graph=True,
    )

    assert float(loss.detach()) < 1e-20
    assert float(metrics["price_normalized_mse"]) < 1e-20
    assert float(metrics["delta_normalized_mse"]) < 1e-20
    assert float(metrics["vega_normalized_mse"]) < 1e-20


def test_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(7)
    model = OptionMLP(hidden_dims=(16, 8))
    inputs = torch.tensor([[100.0, 100.0, 1.0, 0.03, 0.20]])
    expected = model(inputs).detach()

    checkpoint = save_checkpoint(
        tmp_path / "model.pt",
        model,
        metadata={"metrics": {"price_rmse": 1.23}},
    )
    restored, metadata = load_checkpoint(checkpoint)

    assert restored(inputs).detach().numpy() == pytest.approx(expected.numpy())
    assert metadata["metrics"]["price_rmse"] == pytest.approx(1.23)


def test_legacy_state_dict_checkpoint_is_rejected(tmp_path: Path):
    path = tmp_path / "legacy.pth"
    torch.save(OptionMLP().state_dict(), path)
    with pytest.raises(ValueError, match="legacy"):
        load_checkpoint(path)


def test_loss_configuration_validation_and_metadata():
    from losses.differential_loss import loss_metadata

    targets = torch.tensor([1.0, 2.0, 4.0])
    scales = LossScales.from_targets(targets, targets, targets)
    weights = DifferentialLossWeights(price=1.0, delta=0.5, vega=0.25)
    metadata = loss_metadata(scales, weights)

    assert metadata["scales"]["price"] > 0
    assert metadata["weights"]["delta"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="strictly positive"):
        LossScales(price=0.0, delta=1.0, vega=1.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        DifferentialLossWeights(price=1.0, delta=-1.0, vega=1.0)
    with pytest.raises(ValueError, match="at least one"):
        DifferentialLossWeights(price=0.0, delta=0.0, vega=0.0)
