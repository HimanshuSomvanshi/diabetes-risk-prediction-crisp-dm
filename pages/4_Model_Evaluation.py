import plotly.graph_objects as go
import streamlit as st

from utils.common import (
    inject_global_css, page_header, metric_card, footer,
    load_results, TEAL, CORAL, AMBER, SLATE, MUTED,
)

st.set_page_config(page_title="Model Evaluation · Diabetes CRISP-DM", page_icon="📊", layout="wide")
inject_global_css()
results = load_results()
best_model = results["best_model_name"]
tc = results["threshold_comparison"]
pr = results["pr_curve"]

page_header(
    "Phase 5 · Evaluation",
    "Does the model meet the success criteria?",
    f"Evaluated against Phase 1's stated 4:1 FN:FP cost ratio — not against Accuracy or ROC AUC alone.",
)

st.markdown("##### Threshold comparison")
st.caption(f"Best model by Recall: **{best_model}**. Same model, three different decision thresholds.")

c1, c2, c3 = st.columns(3)
for col, key, label, highlight in [
    (c1, "default", "Default (0.50)", False),
    (c2, "f1_optimized", "F1-optimized", False),
    (c3, "cost_weighted", "Cost-weighted (4:1)", True),
]:
    d = tc[key]
    border = TEAL if highlight else "#E5E1DA"
    bg = "#E3F2F2" if highlight else "#FFFFFF"
    col.markdown(
        f"""
        <div style="background:{bg}; border:2px solid {border}; border-radius:8px; padding:1rem 1.1rem;">
            <div style="font-size:0.78rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.06em; font-weight:600;">{label}</div>
            <div style="font-family:'Source Serif 4',serif; font-size:1.6rem; color:{SLATE}; font-weight:700; margin:0.2rem 0;">
                t = {d['threshold']:.3f}
            </div>
            <div style="font-size:0.88rem; line-height:1.6;">
                Precision: <b>{d['precision']:.3f}</b><br>
                Recall: <b>{d['recall']:.3f}</b><br>
                F1: <b>{d['f1']:.3f}</b><br>
                Weighted cost (4·FN+FP): <b>{d['weighted_cost']:.0f}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown(
    f"""
    <div class="callout-teal">
    <b>Why the cost-weighted threshold wins:</b> it minimizes <code>4×FN + 1×FP</code> on
    the test set — weighted cost of {tc['cost_weighted']['weighted_cost']:.0f}, versus
    {tc['default']['weighted_cost']:.0f} at the default threshold. It trades
    {tc['cost_weighted']['fp']-tc['default']['fp']} additional false positives for
    {tc['default']['fn']-tc['cost_weighted']['fn']} fewer false negatives — a trade this
    project's stated cost assumption says is worth making.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.markdown("##### Precision / Recall / F1 across all thresholds")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pr["thresholds"], y=pr["precisions"], name="Precision", line=dict(color=AMBER, width=2)))
fig.add_trace(go.Scatter(x=pr["thresholds"], y=pr["recalls"], name="Recall", line=dict(color=TEAL, width=2)))
fig.add_trace(go.Scatter(x=pr["thresholds"], y=pr["f1_scores"], name="F1", line=dict(color=CORAL, width=2, dash="dash")))
for key, color, label in [("default", MUTED, "Default"), ("f1_optimized", AMBER, "F1-optimal"), ("cost_weighted", CORAL, "Cost-weighted")]:
    fig.add_vline(x=tc[key]["threshold"], line=dict(color=color, dash="dot", width=1.5))
fig.update_layout(
    height=380, margin=dict(t=10, b=10, l=10, r=10),
    xaxis_title="Decision threshold", yaxis_title="Score",
    legend=dict(orientation="h", y=1.12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
)
st.plotly_chart(fig, width="stretch")

st.write("")
left, right = st.columns(2)
with left:
    st.markdown("##### Does this meet the success criteria?")
    cw = tc["cost_weighted"]
    st.markdown(
        f"""
        At the cost-weighted threshold, the model achieves **Recall = {cw['recall']:.4f}**
        and **Precision = {cw['precision']:.4f}** on the held-out test set. Given the
        4× FN:FP cost assumption, **this is an acceptable tradeoff**: the weighted cost
        ({cw['weighted_cost']:.0f}) is lower than both the default ({tc['default']['weighted_cost']:.0f})
        and F1-optimized ({tc['f1_optimized']['weighted_cost']:.0f}) thresholds — fewer
        costly errors overall, even with more false alarms in absolute terms
        ({cw['fp']} of {cw['fp']+cw['tn']} healthy test patients flagged positive).

        Recall alone clears 80% here too, but that's a *consequence* of the cost ratio,
        not a separate justification — a model flagging everyone positive would trivially
        hit 100% Recall while being useless. The cost ratio is what disciplines the
        threshold choice.
        """
    )

with right:
    st.markdown("##### Would this model be shipped?")
    st.markdown(
        """
        **Yes, with conditions.** It meets the stated success criteria on this dataset
        and population. Conditions:

        1. Not assumed to generalize beyond female Pima Indian patients 21+ — untested
           elsewhere.
        2. ~46% precision means roughly half of positive flags are false alarms — fine
           under the stated cost assumption, but should be surfaced to any end user
           (see Live Predictor), not hidden behind a binary label.
        3. Insulin's near-zero feature importance (Phase 6) suggests its heavy
           imputation (~49% of values) didn't materially help this model — worth
           revisiting if the deployed model changes.
        """
    )

st.markdown(
    """
    <div class="callout-amber">
    <b>Limitations:</b> test set is only ~154 rows (CV std in Phase 4 shows how much a
    single split can vary); SkinThickness/Insulin were heavily imputed; no real-world
    deployment context was defined in Phase 1, so "success" is evaluated against the
    stated Recall/cost framing only, not an external business requirement.
    </div>
    """,
    unsafe_allow_html=True,
)

footer()
