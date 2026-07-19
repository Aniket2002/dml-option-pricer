# Differential ML Option Pricer

A reproducible PyTorch surrogate for **no-dividend European call options**. The model is trained on Black-Scholes prices together with analytic delta and vega labels, then evaluated against the same closed-form benchmark on a held-out synthetic test set.

## What is different about this implementation

- **Differential supervision:** the objective includes price, delta and vega errors.
- **Correct derivative units:** feature scaling occurs inside the neural computational graph, so `torch.autograd.grad` returns delta and vega with respect to the original financial inputs. No manual scaler correction is required.
- **Comparable loss terms:** price, delta and vega residuals are divided by training-set scales before weighting.
- **Call-price bounds:** the network output is constrained to
  `max(S - K exp(-rT), 0) <= C <= S`.
- **Reproducible splits and seeds:** data generation, train/validation/test splits and loaders use explicit seeds.
- **Generated evidence:** the training script writes held-out metrics to `artifacts/metrics.json`; the README does not hard-code performance claims.
- **Defensive tests and CI:** analytic Black-Scholes values, raw-unit Greeks, bounds, checkpoint compatibility and training utilities are tested on Python 3.10-3.12.

## Scope

The current model covers:

- European calls;
- no dividends;
- constant volatility and continuously compounded rates;
- inputs `[S, K, T, r, sigma]`;
- outputs price, delta and vega.

It is an educational surrogate-model project, not a market-calibrated pricing system.

## Installation

Tested in CI on Python 3.10, 3.11 and 3.12.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Train

A standard training run:

```bash
python -m train.train_model
```

Train the DML model and a price-only benchmark on the same split:

```bash
python -m train.train_model --compare-baseline
```

Useful development-sized run:

```bash
python -m train.train_model --samples 5000 --epochs 20 --batch-size 256
```

The script creates:

```text
artifacts/dml_option_pricer.pt
artifacts/metrics.json
```

The checkpoint stores the model state, input-scaling statistics, architecture and training metadata. Legacy state-dict-only checkpoints are intentionally rejected because they do not contain the scaling information needed for correct raw-unit Greeks.

## Dashboard

After training:

```bash
streamlit run streamlit_app/app.py
```

The dashboard:

- compares neural price, delta and vega with Black-Scholes values;
- warns when inputs are outside the training domain;
- displays held-out metrics from the checkpoint rather than fixed claims;
- plots price, delta and vega error surfaces;
- exports a model card as JSON.

## Tests and linting

```bash
python -m ruff check .
python -m pytest -q \
  --cov=data \
  --cov=models \
  --cov=losses \
  --cov=train \
  --cov-report=term-missing \
  --cov-fail-under=75
```

## Mathematical conventions

For a no-dividend European call:

```text
d1 = [ln(S/K) + (r + 0.5 sigma^2)T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
C  = S N(d1) - K exp(-rT) N(d2)
Delta = N(d1)
Vega  = S phi(d1) sqrt(T)
```

Vega is reported per **1.00** change in volatility. Divide it by 100 for a one-volatility-point sensitivity.

## Repository layout

```text
dml-option-pricer/
├── data/
│   └── bs_data_generator.py
├── losses/
│   └── differential_loss.py
├── models/
│   └── dml_model.py
├── train/
│   └── train_model.py
├── streamlit_app/
│   └── app.py
├── tests/
├── artifacts/
├── .github/workflows/ci.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Limitations

- Synthetic Black-Scholes supervision is not market validation.
- The model does not handle dividends, American exercise, barriers, stochastic volatility or jumps.
- Accuracy outside the sampled domain is not established.
- The annualized rate and volatility conventions must match the Black-Scholes inputs.
- The no-arbitrage output layer enforces simple call bounds, not every static-arbitrage relationship across an entire surface.

## Suggested CV wording

> Developed a differentiable PyTorch surrogate for European call pricing, using joint price-and-Greek supervision, in-graph feature scaling for raw-unit autograd Greeks, no-arbitrage output bounds, reproducible benchmarking and an interactive error-analysis dashboard.
