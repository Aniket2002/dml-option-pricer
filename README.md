Differential ML Option Pricing Model

````markdown
[![Build Status](https://github.com/yourusername/dml-option-pricer/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/dml-option-pricer/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deploy-url.streamlitapp.com)

# Differential ML Option Pricer

Interactive ML-based pricer with AAD-computed Greeks vs. Black-Scholes baseline.

---

## 📸 Dashboard Preview

![Dashboard Screenshot](docs/dashboard.png)

---

## 🚀 Quick Start

We use **Python 3.10** for full compatibility with all dependencies.

```bash
# 1. Clone
git clone https://github.com/yourusername/dml-option-pricer.git
cd dml-option-pricer

# 2. Install (uses Pipenv for reproducible lockfile)
pip install pipenv
pipenv install --dev        # installs Pipfile.lock exactly
pipenv shell

# 3. Generate data & train (optional if you just want the dashboard)
python data/bs_data_generator.py
python train/train_model.py

# 4. Launch app
streamlit run streamlit_app/app.py --server.fileWatcherType none
````

---

## 🎯 Core Highlights

* **Differential Supervision**
  Trains on both prices and analytic Greeks (Δ & Vega) via a composite loss.

* **Adjoint Algorithmic Differentiation**
  Real-time Δ and Vega computed with `torch.autograd.grad`—no finite differences.

* **Hyperparameter Sweep & Reproducibility**

  * Best config: `lr=1e-3, batch_size=128, λΔ=2.0, λν=0.5`
  * Validation RMSEs: **Price < 1%**, **Δ < 0.03**, **Vega < 1.7**
  * Lockfile (`Pipfile.lock`) ensures exact environments.

* **Interactive Dashboard**

  * **Overview**: One-click ML vs. BSM comparison + error table + bullet takeaways
  * **Deep Analysis**: 2D error heatmap, 1D surface slices, distribution plots, seed-controlled grids

---

## 🗂 Repository Layout

```
dml-option-pricer/
├── data/                   # BSM data generator + augmentation
├── models/                 # Differentiable MLP (OptionMLP)
├── losses/                 # Composite price+Greek loss (AAD)
├── train/                  # Training loop + hyperparameter sweep
├── notebooks/              # Jupyter analyses & visualizations
├── streamlit_app/          # Streamlit dashboard
├── Pipfile
├── Pipfile.lock
├── requirements.txt        # fallback for pip
├── docs/dashboard.png      # dashboard screenshot
├── LICENSE
└── README.md
```

---

## 🔍 Forward Roadmap

* **Benchmark Latency** & optimize model size
* **Integrate Real Market Data** (e.g. live feeds via WebSocket)
* **Deploy to Cloud** (AWS Lambda / Docker / Kubernetes)
* **Add Automated Tests** & CI/CD for model drift monitoring

---

## 📞 Contact

**Aniket Bhardwaj** • [aniket.bhardwaj@domain.com](mailto:bhardwaj.aniket2002@gmail.com) • [LinkedIn](https://www.linkedin.com/in/aniket-bhardwaj-b002/)

```
```
