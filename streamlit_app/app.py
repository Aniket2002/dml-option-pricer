from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.bs_data_generator import (  # noqa: E402
    black_scholes_call_price,
    black_scholes_delta,
    black_scholes_vega,
)
from models.dml_model import load_checkpoint, price_and_greeks  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_ROOT / "artifacts" / "dml_option_pricer.pt"


@st.cache_resource
def cached_checkpoint(path_text: str, modified_ns: int):
    del modified_ns
    return load_checkpoint(path_text, device="cpu")


def load_demo_model(path: Path):
    if not path.exists():
        st.error(
            "No compatible checkpoint was found. Run `python -m train.train_model` "
            "from the repository root, then reload this page."
        )
        st.stop()
    try:
        return cached_checkpoint(str(path), path.stat().st_mtime_ns)
    except (ValueError, RuntimeError) as exc:
        st.error(f"The checkpoint is incompatible: {exc}")
        st.stop()


def model_card(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope": metadata.get("model_scope", "no-dividend European calls"),
        "sampling_bounds": metadata.get("sampling_bounds", {}),
        "test_metrics": metadata.get("metrics", {}),
        "baseline_metrics": metadata.get("baseline_metrics"),
        "loss": metadata.get("loss", {}),
        "limitations": [
            "Black-Scholes synthetic supervision",
            "European calls only",
            "No dividends, early exercise or stochastic volatility",
            "Extrapolation outside the training domain is not validated",
        ],
    }


def in_training_domain(
    values: dict[str, float],
    bounds: dict[str, Any],
) -> bool:
    mapping = {
        "S": "spot",
        "K": "strike",
        "T": "maturity",
        "r": "rate",
        "sigma": "volatility",
    }
    for feature_name, bound_name in mapping.items():
        if bound_name not in bounds:
            continue
        low, high = bounds[bound_name]
        if not low <= values[feature_name] <= high:
            return False
    return True


def predict_single(
    model: torch.nn.Module,
    values: dict[str, float],
) -> dict[str, float]:
    inputs = torch.tensor(
        [[values["S"], values["K"], values["T"], values["r"], values["sigma"]]],
        dtype=torch.float32,
    )
    price, delta, vega = price_and_greeks(model, inputs)
    return {
        "price": float(price.item()),
        "delta": float(delta.item()),
        "vega": float(vega.item()),
    }


def predict_grid(
    model: torch.nn.Module,
    inputs: np.ndarray,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    vegas: list[np.ndarray] = []

    for start in range(0, len(inputs), batch_size):
        batch = torch.tensor(inputs[start : start + batch_size], dtype=torch.float32)
        price, delta, vega = price_and_greeks(model, batch)
        prices.append(price.detach().numpy())
        deltas.append(delta.detach().numpy())
        vegas.append(vega.detach().numpy())

    return (
        np.concatenate(prices),
        np.concatenate(deltas),
        np.concatenate(vegas),
    )


def main() -> None:
    st.set_page_config(
        page_title="Differential ML Option Pricer",
        page_icon="📈",
        layout="wide",
    )
    model, metadata = load_demo_model(DEFAULT_CHECKPOINT)
    card = model_card(metadata)

    st.title("Differential ML Option Pricer")
    st.caption(
        "A bounded neural surrogate for no-dividend European calls, trained on "
        "Black-Scholes prices, deltas and vegas."
    )

    with st.sidebar:
        st.header("Contract inputs")
        spot = st.number_input("Spot (S)", min_value=0.01, value=100.0, step=1.0)
        strike = st.number_input("Strike (K)", min_value=0.01, value=100.0, step=1.0)
        maturity = st.number_input(
            "Maturity in years (T)",
            min_value=0.001,
            value=1.0,
            step=0.05,
            format="%.3f",
        )
        rate = st.number_input(
            "Continuously compounded rate (r)",
            min_value=-0.10,
            max_value=0.30,
            value=0.03,
            step=0.005,
            format="%.3f",
        )
        volatility = st.number_input(
            "Volatility (sigma)",
            min_value=0.001,
            max_value=2.0,
            value=0.20,
            step=0.01,
            format="%.3f",
        )
        st.download_button(
            "Download model card",
            data=json.dumps(card, indent=2),
            file_name="model_card.json",
            mime="application/json",
        )

    values = {
        "S": float(spot),
        "K": float(strike),
        "T": float(maturity),
        "r": float(rate),
        "sigma": float(volatility),
    }
    if not in_training_domain(values, card.get("sampling_bounds", {})):
        st.warning(
            "At least one input is outside the training domain. The neural output "
            "is an extrapolation and should not be treated as validated."
        )

    neural = predict_single(model, values)
    analytic = {
        "price": float(
            black_scholes_call_price(spot, strike, maturity, rate, volatility)
        ),
        "delta": float(
            black_scholes_delta(spot, strike, maturity, rate, volatility)
        ),
        "vega": float(
            black_scholes_vega(spot, strike, maturity, rate, volatility)
        ),
    }

    columns = st.columns(3)
    for column, key, label in zip(
        columns,
        ("price", "delta", "vega"),
        ("Price", "Delta", "Vega"),
        strict=True,
    ):
        difference = neural[key] - analytic[key]
        column.metric(
            label,
            f"{neural[key]:.6f}",
            delta=f"{difference:+.6f} vs BSM",
            delta_color="off",
        )

    comparison = pd.DataFrame(
        {
            "Metric": ["Price", "Delta", "Vega"],
            "Black-Scholes": [analytic["price"], analytic["delta"], analytic["vega"]],
            "Neural": [neural["price"], neural["delta"], neural["vega"]],
        }
    )
    comparison["Absolute error"] = np.abs(
        comparison["Neural"] - comparison["Black-Scholes"]
    )
    comparison["Relative error (%)"] = np.where(
        comparison["Black-Scholes"].abs() > 1e-10,
        100.0 * comparison["Absolute error"] / comparison["Black-Scholes"].abs(),
        np.nan,
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    test_metrics = card.get("test_metrics", {})
    if test_metrics:
        st.subheader("Held-out synthetic test metrics")
        metric_columns = st.columns(3)
        metric_columns[0].metric("Price RMSE", f"{test_metrics.get('price_rmse', math.nan):.6f}")
        metric_columns[1].metric("Delta RMSE", f"{test_metrics.get('delta_rmse', math.nan):.6f}")
        metric_columns[2].metric("Vega RMSE", f"{test_metrics.get('vega_rmse', math.nan):.6f}")

    st.subheader("Error surface")
    surface_columns = st.columns(3)
    spot_min, spot_max = surface_columns[0].slider(
        "Spot range",
        min_value=1.0,
        max_value=300.0,
        value=(50.0, 150.0),
    )
    maturity_min, maturity_max = surface_columns[1].slider(
        "Maturity range",
        min_value=0.01,
        max_value=5.0,
        value=(0.1, 2.0),
    )
    grid_size = surface_columns[2].slider("Grid resolution", 20, 120, 60)

    spot_values = np.linspace(spot_min, spot_max, grid_size)
    maturity_values = np.linspace(maturity_min, maturity_max, grid_size)
    spot_grid, maturity_grid = np.meshgrid(spot_values, maturity_values)
    grid_inputs = np.column_stack(
        [
            spot_grid.ravel(),
            np.full(spot_grid.size, strike),
            maturity_grid.ravel(),
            np.full(spot_grid.size, rate),
            np.full(spot_grid.size, volatility),
        ]
    )
    neural_price, neural_delta, neural_vega = predict_grid(model, grid_inputs)
    analytic_price = black_scholes_call_price(
        grid_inputs[:, 0],
        grid_inputs[:, 1],
        grid_inputs[:, 2],
        grid_inputs[:, 3],
        grid_inputs[:, 4],
    )
    analytic_delta = black_scholes_delta(
        grid_inputs[:, 0],
        grid_inputs[:, 1],
        grid_inputs[:, 2],
        grid_inputs[:, 3],
        grid_inputs[:, 4],
    )
    analytic_vega = black_scholes_vega(
        grid_inputs[:, 0],
        grid_inputs[:, 1],
        grid_inputs[:, 2],
        grid_inputs[:, 3],
        grid_inputs[:, 4],
    )

    surface_tabs = st.tabs(["Price", "Delta", "Vega"])
    surfaces = (
        (surface_tabs[0], np.abs(neural_price - analytic_price), "Absolute price error"),
        (surface_tabs[1], np.abs(neural_delta - analytic_delta), "Absolute delta error"),
        (surface_tabs[2], np.abs(neural_vega - analytic_vega), "Absolute vega error"),
    )
    for tab, error_values, title in surfaces:
        with tab:
            error_grid = error_values.reshape(spot_grid.shape)
            figure = px.imshow(
                error_grid,
                x=spot_values,
                y=maturity_values,
                origin="lower",
                aspect="auto",
                labels={"x": "Spot", "y": "Maturity", "color": "Absolute error"},
                title=title,
            )
            st.plotly_chart(figure, use_container_width=True)

    st.info(
        "This is an educational surrogate-model demonstration. It is not a "
        "market-calibrated pricing library or trading recommendation."
    )


if __name__ == "__main__":
    main()
