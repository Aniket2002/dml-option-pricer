import numpy as np
import pytest

from data.bs_data_generator import (
    SamplingBounds,
    black_scholes_call_price,
    black_scholes_delta,
    black_scholes_theta,
    black_scholes_vega,
    generate_synthetic_data,
)


def test_known_at_the_money_values():
    price = black_scholes_call_price(100.0, 100.0, 1.0, 0.05, 0.20)
    delta = black_scholes_delta(100.0, 100.0, 1.0, 0.05, 0.20)
    vega = black_scholes_vega(100.0, 100.0, 1.0, 0.05, 0.20)
    theta = black_scholes_theta(100.0, 100.0, 1.0, 0.05, 0.20)

    assert float(price) == pytest.approx(10.4505835722, rel=1e-9)
    assert float(delta) == pytest.approx(0.6368306512, rel=1e-9)
    assert float(vega) == pytest.approx(37.5240346917, rel=1e-9)
    assert float(theta) == pytest.approx(-6.4140275464, rel=1e-9)


def test_vectorized_outputs_and_no_arbitrage_bounds():
    spot = np.array([80.0, 100.0, 120.0])
    strike = 100.0
    maturity = 1.0
    rate = 0.03
    volatility = 0.20
    price = black_scholes_call_price(spot, strike, maturity, rate, volatility)
    lower = np.maximum(spot - strike * np.exp(-rate * maturity), 0.0)

    assert price.shape == (3,)
    assert np.all(price >= lower)
    assert np.all(price <= spot)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="maturity"):
        black_scholes_call_price(100.0, 100.0, 0.0, 0.03, 0.20)
    with pytest.raises(ValueError, match="volatility"):
        black_scholes_call_price(100.0, 100.0, 1.0, 0.03, 0.0)


def test_generator_is_reproducible_and_stays_in_domain():
    bounds = SamplingBounds()
    first = generate_synthetic_data(100, seed=7, bounds=bounds, augment=True)
    second = generate_synthetic_data(100, seed=7, bounds=bounds, augment=True)

    assert len(first) == 200
    assert first.equals(second)
    assert first["S"].between(*bounds.spot).all()
    assert first["K"].between(*bounds.strike).all()
    assert first["T"].between(*bounds.maturity).all()
    assert first["sigma"].between(*bounds.volatility).all()
