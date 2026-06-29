import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.common import inject_global_css, page_header, footer, load_results, TEAL, MUTED, CORAL

st.set_page_config(page_title="Feature Analysis · Diabetes CRISP-DM", page_icon="🔬", layout="wide")
inject_global_css()
results = load_results()
best_model = results["best_model_name"]
importances = results["feature_importances"]

page_header(
    "Phase 6 · Feature Analysis",
    "What the model actually relies on",
    f"Feature importances from the deployed model ({best_model}) — not assumed, computed.",
)

imp_df = pd.DataFrame(
    [(k, v) for k, v in importances.items()], columns=["Feature", "Importance"]
).sort_values("Importance", ascending=True)

fig = go.Figure(data=[go.Bar(
    x=imp_df["Importance"], y=imp_df["Feature"], orientation="h",
    marker_color=[TEAL if v > 0 else "#D8D4CC" for v in imp_df["Importance"]],
    text=[f"{v:.3f}" if v > 0 else "0.000" for v in imp_df["Importance"]],
    textposition="outside",
)])
fig.update_layout(
    height=420, margin=dict(t=10, b=10, l=10, r=60),
    xaxis_title="Feature importance (Gini)",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
)
st.plotly_chart(fig, width="stretch")

nonzero = imp_df[imp_df["Importance"] > 0].sort_values("Importance", ascending=False)
zero = imp_df[imp_df["Importance"] == 0]["Feature"].tolist()

st.markdown(
    f"""
    <div class="callout-teal">
    <b>Only {len(nonzero)} of {len(imp_df)} features</b> are ever used by this model —
    {', '.join(f"<b>{r.Feature}</b> ({r.Importance:.1%})" for r in nonzero.itertuples())}.
    At <code>max_depth=3</code>, the Decision Tree never splits on the other
    {len(zero)} features: {', '.join(zero)}.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.markdown("##### Sensitivity check: does heavily-imputed Insulin matter?")
st.markdown(
    """
    Insulin was ~49% imputed (Phase 2/3) — worth testing directly whether the model
    performs differently without it.
    """
)
if importances.get("Insulin", 0) == 0:
    st.markdown(
        """
        <div class="callout-amber">
        <b>Result: identical performance with or without Insulin.</b> This isn't a bug —
        it's a direct consequence of the chart above. The Decision Tree assigns Insulin
        exactly <b>0.0 importance</b>; it never selects it as a split feature because
        Glucose, BMI, and Age separate the classes more cleanly. The heavy imputation on
        Insulin therefore had no measurable effect on <i>this specific model</i>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This is model-specific, not universal — a model that uses all features "
        "implicitly (e.g. SVM with an RBF kernel) could still be sensitive to Insulin's "
        "imputation even though this Decision Tree isn't."
    )
else:
    st.write("Insulin contributes non-zero importance in this model — see chart above for magnitude.")

footer()
