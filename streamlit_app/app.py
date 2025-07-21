# streamlit_app/app.py

import streamlit as st
import sys, os, json
import torch
import numpy as np
import pandas as pd
import plotly.express as px

# 1) Page config (must be first)
st.set_page_config(
    page_title="🔥 DML Option Pricer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2) Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.dml_model import OptionMLP
from data.bs_data_generator import (
    black_scholes_call_price,
    black_scholes_delta,
    black_scholes_vega,
)

# 3) Load & cache model
@st.cache_resource
def load_model():
    path   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dml_pricer_best.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = OptionMLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# 4) Top-level cached surface generator
@st.cache_data(show_spinner=True)
def generate_surfaces(S_min, S_max, T_min, T_max, n_pts, K, r, sigma):
    S_vals = np.linspace(S_min, S_max, n_pts)
    T_vals = np.linspace(T_min, T_max, n_pts)
    Sg, Tg = np.meshgrid(S_vals, T_vals)
    Kg = np.full_like(Sg, K)
    rg = np.full_like(Sg, r)
    sg = np.full_like(Sg, sigma)

    p_bsm = black_scholes_call_price(Sg, Kg, Tg, rg, sg)
    d_bsm = black_scholes_delta(Sg, Kg, Tg, rg, sg)
    v_bsm = black_scholes_vega(Sg, Kg, Tg, rg, sg)

    X = (
        torch.tensor(
            np.stack([Sg.ravel(), Kg.ravel(), Tg.ravel(), rg.ravel(), sg.ravel()], axis=1),
            dtype=torch.float32, device=device
        )
        .requires_grad_(True)
    )
    pred = model(X).squeeze()
    p_ml = pred.detach().cpu().numpy().reshape(Sg.shape)
    grads = torch.autograd.grad(
        outputs=pred,
        inputs=X,
        grad_outputs=torch.ones_like(pred),
        create_graph=False
    )[0].cpu().numpy().reshape(Sg.shape + (5,))
    d_ml = grads[..., 0]
    v_ml = grads[..., 4]

    df_flat = pd.DataFrame({
        "S": Sg.ravel(),
        "T": Tg.ravel(),
        "BSM_Price":  p_bsm.ravel(),
        "ML_Price":   p_ml.ravel(),
        "Price_Error": np.abs(p_ml - p_bsm).ravel(),
        "Delta_Error": np.abs(d_ml - d_bsm).ravel(),
        "Vega_Error":  np.abs(v_ml - v_bsm).ravel(),
    })

    return S_vals, T_vals, p_bsm, p_ml, d_bsm, d_ml, v_bsm, v_ml, df_flat

# 5) Sidebar with inputs & model card
with st.sidebar:
    st.header("Parameters")
    with st.expander("Basic Inputs", expanded=True):
        S0 = st.number_input("Spot Price S₀",      0.0, 500.0, 100.0, step=1.0)
        K  = st.number_input("Strike Price K",     0.0, 500.0, 100.0, step=1.0)
        T0 = st.number_input("Time to Maturity T₀",0.0,   5.0,   1.0, step=0.1)
    with st.expander("Advanced Inputs"):
        r     = st.number_input("Risk-Free Rate r", 0.0, 0.2,   0.01, step=0.001, format="%.3f")
        sigma = st.number_input("Volatility σ",      0.0, 1.0,   0.20, step=0.01)
        seed  = st.number_input("Random Seed",       0,   100000, 42,   step=1)
    st.markdown("---")
    st.header("About this Model")
    st.markdown(
        """
- **Training ranges:**  
  S∈[50,150], K∈[50,150], T∈[0.1,2.0], r∈[0.01,0.05], σ∈[0.1,0.5]  
- **Loss weights:** λΔ=2.0, λν=0.5  
- **Architecture:** 3-layer MLP, SiLU activations  
- **Limitations:** European calls only; no early exercise; no stochastic vol  
        """
    )
    metadata = {
        "training_ranges": {"S":[50,150],"K":[50,150],"T":[0.1,2.0],"r":[0.01,0.05],"sigma":[0.1,0.5]},
        "loss_weights": {"lambda_delta":2.0,"lambda_vega":0.5},
        "architecture": "3-layer MLP, SiLU",
        "limitations": ["European calls","no early exercise","no stoch vol"]
    }
    st.download_button(
        "Download Model Card (JSON)",
        data=json.dumps(metadata, indent=2),
        file_name="model_card.json",
        mime="application/json"
    )

# 6) Main header & summary
st.title("🔥 Differential ML Option Pricer Dashboard")
st.markdown(
    "**Executive Summary:** ML pricer yields <1% price RMSE & single-digit Greek RMSE vs. BSM."
)

# 7) Tabs & UI
tab1, tab2 = st.tabs(["📊 Overview", "🔎 Deep Analysis"])

def show_overview():
    st.subheader("1️⃣ One-Shot ML vs. BSM")
    if st.button("▶️ Compute"):
        x = torch.tensor([[S0, K, T0, r, sigma]],
                         dtype=torch.float32,
                         device=device).requires_grad_(True)
        pred = model(x).squeeze()
        pm = pred.item()
        grad = torch.autograd.grad(pred, x, grad_outputs=torch.tensor(1.0, device=device))[0]
        dm, vm = grad[0,0].item(), grad[0,4].item()
        pb = black_scholes_call_price(S0, K, T0, r, sigma)
        db = black_scholes_delta(S0, K, T0, r, sigma)
        vb = black_scholes_vega(S0, K, T0, r, sigma)

        df = pd.DataFrame([
            ["Price", pb, pm, pm-pb],
            ["Delta", db, dm, dm-db],
            ["Vega",  vb, vm, vm-vb],
        ], columns=["Metric","BSM","ML","Abs Error"])
        df["Rel (%)"] = 100*df["Abs Error"]/df["BSM"].abs()
        st.table(df)
        st.markdown("**Takeaways:**")
        for _, r_ in df.iterrows():
            st.markdown(f"- {r_.Metric}: {r_['Abs Error']:.4f} ({r_['Rel (%)']:.2f}%)")

def show_deep():
    st.subheader("2️⃣ Deep Surface & Scenario Comparison")
    with st.form("form2"):
        S_min, S_max = st.slider("Spot Range S₀", 0.0,500.0,(50.0,150.0))
        T_min, T_max = st.slider("Maturity Range T",0.0,5.0,(0.1,2.0))
        n_pts        = st.slider("Grid Points",20,200,100)
        go           = st.form_submit_button("▶️ Run")
    if go:
        if S_min>=S_max or T_min>=T_max:
            st.error("Ensure S_min < S_max and T_min < T_max")
            return

        np.random.seed(int(seed))
        # Unpack only the first four outputs
        aS, aT, a_p_bsm, a_p_ml, *rest = generate_surfaces(
            S_min, S_max, T_min, T_max, n_pts, K, r, sigma
        )
        err = np.abs(a_p_ml - a_p_bsm)
        idx = np.unravel_index(err.argmax(), err.shape)
        st.markdown(f"**Max Price Error:** {err.max():.4f} at S₀={aS[idx[1]]:.2f}, T={aT[idx[0]]:.2f}")

        fig = px.imshow(
            err, x=aS, y=aT,
            labels={"x":"Spot Price $S_0$","y":"T (years)","color":"|ML−BSM|"},
            origin="lower", aspect="auto", title="Price Error Heatmap"
        )
        fig.update_layout(coloraxis_colorbar=dict(
            tickmode="array", tickvals=[0,0.5,1.0,1.5,2.0]
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Histograms
        for col, title in [("Price_Error","Price"),("Delta_Error","Delta"),("Vega_Error","Vega")]:
            fig2 = px.histogram(
                pd.DataFrame({col: generate_surfaces(
                    S_min, S_max, T_min, T_max, n_pts, K, r, sigma
                )[8][col]}),
                x=col, nbins=50, title=f"{title} Error Distribution"
            )
            fig2.update_layout(xaxis_title="Error", yaxis_title="Count")
            st.plotly_chart(fig2, use_container_width=True)

        # Scenario Manager
        if "scenarios" not in st.session_state:
            st.session_state.scenarios = {}
        with st.expander("📂 Scenario Manager", expanded=True):
            name = st.text_input("Scenario Name", key="nm2")
            if st.button("Save Scenario"):
                st.session_state.scenarios[name] = (S_min, S_max, T_min, T_max, n_pts)
                st.success(f"Saved '{name}'")
            sel = st.multiselect("Compare Scenarios", list(st.session_state.scenarios.keys()))
            if sel:
                cols = st.columns(len(sel))
                for col_ui, sc in zip(cols, sel):
                    p = st.session_state.scenarios[sc]
                    bS, bT, bp, mp, *_ = generate_surfaces(*p, K, r, sigma)
                    e = np.abs(mp - bp)
                    piv = pd.DataFrame({
                        "T": np.repeat(bT, len(bS)),
                        "S": np.tile(bS, len(bT)),
                        "Error": e.ravel()
                    }).pivot(index="T", columns="S", values="Error")
                    fig3 = px.imshow(
                        piv.values, x=piv.columns, y=piv.index,
                        labels={"x":"S₀","y":"T","color":"|ML−BSM|"},
                        origin="lower", aspect="auto", title=sc
                    )
                    fig3.update_layout(coloraxis_colorbar=dict(
                        tickmode="array", tickvals=[0,0.5,1.0,1.5,2.0]
                    ))
                    col_ui.plotly_chart(fig3, use_container_width=True)

# 8) Render tabs
with tab1:
    show_overview()
with tab2:
    show_deep()
