import streamlit as st

from utils.common import (
    inject_global_css, page_header, metric_card, footer,
    load_results, SLATE, TEAL, MUTED,
)

st.set_page_config(
    page_title="Diabetes Risk Prediction · CRISP-DM",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
results = load_results()

with st.sidebar:
    st.markdown("### 🩺 Diabetes Prediction")
    st.caption("PIMA Dataset · HTW Berlin")
    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:0.85rem; line-height:1.6;">
        <b>Model:</b> {results['best_model_name']}<br>
        <b>Primary metric:</b> Recall<br>
        <b>Cost assumption:</b> FN = 4× FP<br>
        <b>Validation:</b> 5-fold Stratified CV<br>
        <b>Methodology:</b> CRISP-DM
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("· MSc Data Science & Project Management · 2026")

page_header(
    "Phase 1 · Business Understanding",
    "Diabetes Risk Prediction",
    "A CRISP-DM walkthrough of a binary classification project on the PIMA Indians Diabetes dataset.",
)

col1, col2, col3, col4 = st.columns(4)
metric_card("Patients", f"{results['dataset']['n_rows']}", col1)
metric_card("Features", f"{results['dataset']['n_features']}", col2)
metric_card("Diabetic Rate", f"{results['dataset']['outcome_pct']['1']:.1f}%", col3)
metric_card("Best Model Recall", f"{results['model_results'][0]['recall']*100:.1f}%", col4)

st.write("")
left, right = st.columns([3, 2])

with left:
    st.markdown("#### Project scope")
    st.markdown(
        """
        This is a **university project** scoped against the PIMA Indians Diabetes dataset
        only. No specific real-world deployment context (a particular clinic, app, or
        end-user workflow) was defined — that scoping decision is stated explicitly here
        rather than implied.

        **Success criteria:** maximize the chance of correctly identifying true diabetic
        cases (high Recall on the positive class), under the assumption below.
        """
    )
    st.markdown(
        f"""
        <div class="callout-amber">
        <b>Cost assumption (stated, not derived):</b> a False Negative — predicting
        "not diabetic" for someone who is — is treated as <b>4× costlier</b> than a False
        Positive. This ratio drives the metric choice (Recall) and the decision threshold
        used on the Live Predictor page, computed in Phase 4.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("#### How to read this app")
    st.markdown(
        """
        <div class="phase-card"><b>Data Understanding</b><br>
        <span style="color:#6B7280; font-size:0.88rem;">What's actually in the data, and what's wrong with it.</span></div>
        <div class="phase-card"><b>Data Preparation</b><br>
        <span style="color:#6B7280; font-size:0.88rem;">Leakage-safe cleaning, imputation, scaling.</span></div>
        <div class="phase-card"><b>Modeling</b><br>
        <span style="color:#6B7280; font-size:0.88rem;">Five tuned, cross-validated classifiers.</span></div>
        <div class="phase-card"><b>Model Evaluation</b><br>
        <span style="color:#6B7280; font-size:0.88rem;">Cost-weighted threshold, not just accuracy.</span></div>
        <div class="phase-card"><b>Feature Analysis</b><br>
        <span style="color:#6B7280; font-size:0.88rem;">What the model actually relies on.</span></div>
        <div class="phase-card"><b>Live Predictor</b><br>
        <span style="color:#6B7280; font-size:0.88rem;">Try it — with the tradeoffs shown, not hidden.</span></div>
        """,
        unsafe_allow_html=True,
    )

footer()
