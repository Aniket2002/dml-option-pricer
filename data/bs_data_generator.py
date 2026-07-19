from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class SamplingBounds:
    """Sampling domain for the no-dividend European-call dataset."""

    spot: tuple[float, float] = (50.0, 150.0)
    strike: tuple[float, float] = (50.0, 150.0)
    maturity: tuple[float, float] = (0.1, 2.0)
    rate: tuple[float, float] = (0.01, 0.05)
    volatility: tuple[float, float] = (0.1, 0.5)

    def validate(self) -> None:
        positive_fields = {
            "spot": self.spot,
            "strike": self.strike,
            "maturity": self.maturity,
            "volatility": self.volatility,
        }
        for name, bounds in positive_fields.items():
            low, high = bounds
            if low <= 0 or high <= low:
                raise ValueError(f"{name} bounds must satisfy 0 < low < high")

        rate_low, rate_high = self.rate
        if rate_high <= rate_low:
            raise ValueError("rate bounds must satisfy low < high")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _validate_black_scholes_inputs(
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    volatility: np.ndarray,
) -> None:
    if np.any(spot <= 0):
        raise ValueError("spot must be strictly positive")
    if np.any(strike <= 0):
        raise ValueError("strike must be strictly positive")
    if np.any(maturity <= 0):
        raise ValueError("maturity must be strictly positive")
    if np.any(volatility <= 0):
        raise ValueError("volatility must be strictly positive")


def _d1_d2(
    spot: Any,
    strike: Any,
    maturity: Any,
    rate: Any,
    volatility: Any,
) -> tuple[np.ndarray, np.ndarray]:
    spot_array = _as_float_array(spot)
    strike_array = _as_float_array(strike)
    maturity_array = _as_float_array(maturity)
    rate_array = _as_float_array(rate)
    volatility_array = _as_float_array(volatility)

    spot_array, strike_array, maturity_array, rate_array, volatility_array = np.broadcast_arrays(
        spot_array,
        strike_array,
        maturity_array,
        rate_array,
        volatility_array,
    )
    _validate_black_scholes_inputs(
        spot_array,
        strike_array,
        maturity_array,
        volatility_array,
    )

    root_t = np.sqrt(maturity_array)
    d1 = (
        np.log(spot_array / strike_array)
        + (rate_array + 0.5 * volatility_array**2) * maturity_array
    ) / (volatility_array * root_t)
    d2 = d1 - volatility_array * root_t
    return d1, d2


def black_scholes_call_price(
    spot: Any,
    strike: Any,
    maturity: Any,
    rate: Any,
    volatility: Any,
) -> np.ndarray:
    """No-dividend Black-Scholes price for a European call."""

    d1, d2 = _d1_d2(spot, strike, maturity, rate, volatility)
    spot_array = _as_float_array(spot)
    strike_array = _as_float_array(strike)
    maturity_array = _as_float_array(maturity)
    rate_array = _as_float_array(rate)
    return spot_array * norm.cdf(d1) - strike_array * np.exp(-rate_array * maturity_array) * norm.cdf(d2)


def black_scholes_delta(
    spot: Any,
    strike: Any,
    maturity: Any,
    rate: Any,
    volatility: Any,
) -> np.ndarray:
    """Spot delta of a no-dividend European call."""

    d1, _ = _d1_d2(spot, strike, maturity, rate, volatility)
    return norm.cdf(d1)


def black_scholes_vega(
    spot: Any,
    strike: Any,
    maturity: Any,
    rate: Any,
    volatility: Any,
) -> np.ndarray:
    """Vega per unit change in volatility, not per one volatility point."""

    d1, _ = _d1_d2(spot, strike, maturity, rate, volatility)
    spot_array = _as_float_array(spot)
    maturity_array = _as_float_array(maturity)
    return spot_array * norm.pdf(d1) * np.sqrt(maturity_array)


def black_scholes_gamma(
    spot: Any,
    strike: Any,
    maturity: Any,
    rate: Any,
    volatility: Any,
) -> np.ndarray:
    d1, _ = _d1_d2(spot, strike, maturity, rate, volatility)
    spot_array = _as_float_array(spot)
    maturity_array = _as_float_array(maturity)
    volatility_array = _as_float_array(volatility)
    return norm.pdf(d1) / (spot_array * volatility_array * np.sqrt(maturity_array))


def black_scholes_theta(
    spot: Any,
    strike: Any,
    maturity: Any,
    rate: Any,
    volatility: Any,
) -> np.ndarray:
    """Calendar-time theta per year for a no-dividend European call."""

    d1, d2 = _d1_d2(spot, strike, maturity, rate, volatility)
    spot_array = _as_float_array(spot)
    strike_array = _as_float_array(strike)
    maturity_array = _as_float_array(maturity)
    rate_array = _as_float_array(rate)
    volatility_array = _as_float_array(volatility)

    diffusion_term = -(
        spot_array
        * norm.pdf(d1)
        * volatility_array
        / (2.0 * np.sqrt(maturity_array))
    )
    carry_term = -(
        rate_array
        * strike_array
        * np.exp(-rate_array * maturity_array)
        * norm.cdf(d2)
    )
    return diffusion_term + carry_term


def generate_synthetic_data(
    n_samples: int = 20_000,
    seed: int = 42,
    bounds: SamplingBounds | None = None,
    augment: bool = False,
    noise_std: float = 0.01,
) -> pd.DataFrame:
    """Generate deterministic Black-Scholes prices and raw-unit Greeks.

    The optional augmentation applies bounded local perturbations and then
    recomputes all labels analytically. It never perturbs labels directly.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")

    bounds = bounds or SamplingBounds()
    bounds.validate()
    rng = np.random.default_rng(seed)

    spot = rng.uniform(*bounds.spot, n_samples)
    strike = rng.uniform(*bounds.strike, n_samples)
    maturity = rng.uniform(*bounds.maturity, n_samples)
    rate = rng.uniform(*bounds.rate, n_samples)
    volatility = rng.uniform(*bounds.volatility, n_samples)

    features = np.column_stack([spot, strike, maturity, rate, volatility])

    if augment:
        multiplicative_noise = rng.normal(0.0, noise_std, size=(n_samples, 4))
        augmented = features.copy()
        augmented[:, 0] *= np.exp(multiplicative_noise[:, 0])
        augmented[:, 1] *= np.exp(multiplicative_noise[:, 1])
        augmented[:, 2] *= np.exp(multiplicative_noise[:, 2])
        augmented[:, 4] *= np.exp(multiplicative_noise[:, 3])
        augmented[:, 3] += rng.normal(
            0.0,
            noise_std * max(bounds.rate[1] - bounds.rate[0], 1e-8),
            size=n_samples,
        )

        lower = np.array(
            [
                bounds.spot[0],
                bounds.strike[0],
                bounds.maturity[0],
                bounds.rate[0],
                bounds.volatility[0],
            ],
            dtype=np.float64,
        )
        upper = np.array(
            [
                bounds.spot[1],
                bounds.strike[1],
                bounds.maturity[1],
                bounds.rate[1],
                bounds.volatility[1],
            ],
            dtype=np.float64,
        )
        augmented = np.clip(augmented, lower, upper)
        features = np.vstack([features, augmented])

    spot, strike, maturity, rate, volatility = features.T
    frame = pd.DataFrame(
        {
            "S": spot,
            "K": strike,
            "T": maturity,
            "r": rate,
            "sigma": volatility,
            "price": black_scholes_call_price(
                spot,
                strike,
                maturity,
                rate,
                volatility,
            ),
            "delta": black_scholes_delta(
                spot,
                strike,
                maturity,
                rate,
                volatility,
            ),
            "vega": black_scholes_vega(
                spot,
                strike,
                maturity,
                rate,
                volatility,
            ),
        }
    )
    return frame


def save_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":  # pragma: no cover
    dataset = generate_synthetic_data(n_samples=20_000, seed=42)
    destination = save_dataset(dataset, "data/option_data.csv")
    print(f"Saved {len(dataset):,} rows to {destination}")
