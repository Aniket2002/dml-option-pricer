# streamlit_app/app.py

import streamlit as st
# ─ Must be the first Streamlit command ─────────────────────────────────
st.set_page_config(page_title="🔥 DML Option Pricer", layout="wide")

import sys, os
import torch
import numpy as np
import matplotlib.pyplot as plt

# ─ Add project root to PYTHONPATH so you can import your modules ─────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dml_model import OptionMLP
from data.bs_data_generator import (
    black_scholes_call_price,
    black_scholes_delta,
    black_scholes_vega
)

@st.cache_resource
def load_model():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dml_pricer_best.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptionMLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

st.title("🔥 Differential ML Option Pricer")
st.write("Real-time Price & Greeks + Deep Error Analysis")

# ─ Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Input Parameters")
S0    = st.sidebar.number_input("Spot Price S₀",               0.0, 500.0, 100.0, step=1.0)
K     = st.sidebar.number_input("Strike K",                    0.0, 500.0, 100.0, step=1.0)
r     = st.sidebar.number_input("Risk-Free Rate r",            0.0,   0.2,   0.01, step=0.001, format="%.3f")
sigma = st.sidebar.number_input("Volatility σ",                0.0,   1.0,    0.2, step=0.01)
T0    = st.sidebar.number_input("Time to Maturity T₀ (years)", 0.0,   5.0,    1.0, step=0.1)

tab1, tab2 = st.tabs(["Single Calculation", "Deep Error Analysis"])

with tab1:
    st.subheader("ML vs. BSM: One-Shot Calculation")
    if st.button("Compute"):
        x = torch.tensor([[S0, K, T0, r, sigma]], dtype=torch.float32, device=device, requires_grad=True)
        pred = model(x).squeeze()
        price_ml = pred.item()
        grads = torch.autograd.grad(outputs=pred, inputs=x, grad_outputs=torch.tensor(1.0, device=device))[0]
        delta_ml, _, _, _, vega_ml = grads[0].cpu().numpy()

        price_bsm = black_scholes_call_price(S0, K, T0, r, sigma)
        delta_bsm = black_scholes_delta(S0, K, T0, r, sigma)
        vega_bsm  = black_scholes_vega(S0, K, T0, r, sigma)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ML Model**")
            st.write(f"• Price: {price_ml:.4f}")
            st.write(f"• Delta: {delta_ml:.4f}")
            st.write(f"• Vega : {vega_ml:.4f}")
        with c2:
            st.markdown("**BSM Analytic**")
            st.write(f"• Price: {price_bsm:.4f}")
            st.write(f"• Delta: {delta_bsm:.4f}")
            st.write(f"• Vega : {vega_bsm:.4f}")

with tab2:
    st.subheader("Deep Error Analysis")
    st.markdown("▶️ **1D Surface Comparison** and  ▶️ **2D Error Heatmap**")

    # Panel for 1D surfaces
    with st.expander("1D Price & Greeks Surfaces"):
        S_min, S_max = st.slider("Spot Range (S)", 0.0, 500.0, (50.0, 150.0), 10.0)
        n_pts = st.slider("Points per Curve", 50, 500, 200)
        S_vals = np.linspace(S_min, S_max, n_pts)
        K_vals = np.full(n_pts, K)
        T_vals = np.full(n_pts, T0)
        r_vals = np.full(n_pts, r)
        sig_vals = np.full(n_pts, sigma)

        # BSM curves
        p_bsm = black_scholes_call_price(S_vals, K_vals, T_vals, r_vals, sig_vals)
        d_bsm = black_scholes_delta(S_vals, K_vals, T_vals, r_vals, sig_vals)
        v_bsm = black_scholes_vega(S_vals, K_vals, T_vals, r_vals, sig_vals)

        # ML curves
        X = torch.tensor(np.stack([S_vals, K_vals, T_vals, r_vals, sig_vals], axis=1),
                         dtype=torch.float32, device=device, requires_grad=True)
        p_ml = model(X).cpu().detach().numpy()
        grads = torch.autograd.grad(outputs=model(X).squeeze(),
                                    inputs=X,
                                    grad_outputs=torch.ones(n_pts, device=device))[0].cpu().numpy()
        d_ml = grads[:,0]
        v_ml = grads[:,4]

        fig, axes = plt.subplots(3, 1, figsize=(6,12))
        axes[0].plot(S_vals, p_bsm, '--', label="BSM"); axes[0].plot(S_vals, p_ml, label="ML")
        axes[0].set_ylabel("Price"); axes[0].legend()
        axes[1].plot(S_vals, d_bsm, '--', label="BSM"); axes[1].plot(S_vals, d_ml, label="ML")
        axes[1].set_ylabel("Delta"); axes[1].legend()
        axes[2].plot(S_vals, v_bsm, '--', label="BSM"); axes[2].plot(S_vals, v_ml, label="ML")
        axes[2].set_xlabel("Spot Price"); axes[2].set_ylabel("Vega"); axes[2].legend()
        st.pyplot(fig)

    # Panel for 2D error heatmap
    with st.expander("2D Error Heatmap (Price)"):
        S_grid = st.slider("Spot axis range", 0.0, 500.0, (50.0, 200.0), step=10.0)
        T_grid = st.slider("Maturity axis range", 0.1, 5.0, (0.1, 2.0), step=0.1)
        S_vals = np.linspace(S_grid[0], S_grid[1], 100)
        T_vals = np.linspace(T_grid[0], T_grid[1], 100)
        Sg, Tg = np.meshgrid(S_vals, T_vals)
        Kg = np.full_like(Sg, K)
        rg = np.full_like(Sg, r)
        sg = np.full_like(Sg, sigma)

        # compute BSM & ML prices on grid
        p_bsm2 = black_scholes_call_price(Sg, Kg, Tg, rg, sg)
        X2 = torch.tensor(np.stack([Sg.ravel(), Kg.ravel(), Tg.ravel(), rg.ravel(), sg.ravel()], axis=1),
                          dtype=torch.float32, device=device, requires_grad=False)
        p_ml2 = model(X2).cpu().detach().numpy().reshape(Sg.shape)

        err = np.abs(p_ml2 - p_bsm2)

        fig2, ax2 = plt.subplots(figsize=(6,5))
        im = ax2.imshow(err, origin='lower',
                        extent=(S_vals.min(), S_vals.max(), T_vals.min(), T_vals.max()),
                        aspect='auto')
        ax2.set_xlabel("Spot Price S"); ax2.set_ylabel("Time to Maturity T")
        ax2.set_title("Absolute Price Error Heatmap")
        fig2.colorbar(im, ax=ax2, label="|ML−BSM|")
        st.pyplot(fig2)
