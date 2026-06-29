import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.common import (
    inject_global_css, page_header, metric_card, footer,
    load_results, TEAL, CORAL, AMBER, SLATE, MUTED,
)

st.set_page_config(page_title="Modeling · Diabetes CRISP-DM", page_icon="⚙️", layout="wide")
inject_global_css()
results = load_results()

page_header(
    "Phase 4 · Modeling",
    "Five tuned, cross-validated classifiers",
    "GridSearchCV + 5-fold Stratified CV for every model — not sklearn defaults.",
)

cv_df = pd.DataFrame(results["cv_summary"])
model_df = pd.DataFrame(results["model_results"])

st.markdown("##### Cross-validation results (training split, 5-fold stratified)")
cv_display = cv_df[["model", "cv_roc_auc_mean", "cv_recall_mean", "cv_recall_std", "cv_f1_mean", "cv_f1_std"]].copy()
cv_display["CV ROC AUC"] = cv_display["cv_roc_auc_mean"].map(lambda x: f"{x:.3f}")
cv_display["CV Recall"] = cv_display.apply(lambda r: f"{r['cv_recall_mean']:.3f} ± {r['cv_recall_std']:.3f}", axis=1)
cv_display["CV F1"] = cv_display.apply(lambda r: f"{r['cv_f1_mean']:.3f} ± {r['cv_f1_std']:.3f}", axis=1)
st.dataframe(
    cv_display[["model", "CV ROC AUC", "CV Recall", "CV F1"]].rename(columns={"model": "Model"}),
    hide_index=True, width="stretch",
)
st.caption(
    "Mean ± std across 5 folds. Wide std relative to between-model gaps is a sign the "
    "single test-set ranking below shouldn't be over-trusted on its own."
)

st.markdown("##### Best hyperparameters found (GridSearchCV)")
param_cols = st.columns(5)
for i, row in cv_df.iterrows():
    with param_cols[i % 5]:
        st.markdown(f"**{row['model']}**")
        for k, v in row["best_params"].items():
            st.caption(f"`{k}` = {v}")

st.write("")
st.markdown("##### Test-set comparison, sorted by Recall")
st.caption("Recall is this project's primary metric (Phase 1) — default 0.5 threshold shown here; the cost-weighted threshold is computed in Phase 5.")

model_df_sorted = model_df.sort_values("recall", ascending=False)
fig = go.Figure()
metrics_to_plot = [("recall", "Recall", TEAL), ("precision", "Precision", AMBER), ("f1", "F1", CORAL), ("roc_auc", "ROC AUC", SLATE)]
for col, label, color in metrics_to_plot:
    fig.add_trace(go.Bar(name=label, x=model_df_sorted["model"], y=model_df_sorted[col], marker_color=color))
fig.update_layout(
    barmode="group", height=380, margin=dict(t=10, b=10, l=10, r=10),
    yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=1.12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
)
st.plotly_chart(fig, width="stretch")

display_df = model_df_sorted.rename(columns={
    "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
    "recall": "Recall", "f1": "F1", "roc_auc": "ROC AUC",
})[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]]
st.dataframe(
    display_df.style.format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]}),
    hide_index=True, width="stretch",
)

st.write("")
st.markdown("##### Soft-voting ensemble")
ens = results["ensemble_result"]
st.markdown(
    f"""
    <div class="callout-teal">
    Top 3 models by Recall — <b>{", ".join(ens['base_models'])}</b> — combined via soft
    voting. Result: Recall={ens['recall']:.3f}, Precision={ens['precision']:.3f},
    F1={ens['f1']:.3f}, ROC AUC={ens['roc_auc']:.3f}.
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"The ensemble didn't outperform the single best model ({results['best_model_name']}) "
    "on Recall here — included for completeness, not selected as the final deployed model."
)

footer()
