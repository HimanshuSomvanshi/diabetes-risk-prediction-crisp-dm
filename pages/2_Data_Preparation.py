import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.common import (
    inject_global_css, page_header, footer, load_results,
    TEAL, AMBER, CORAL, SLATE, MUTED,
)

st.set_page_config(page_title="Data Preparation · Diabetes CRISP-DM", page_icon="🛠️", layout="wide")
inject_global_css()
results = load_results()

page_header(
    "Phase 3 · Data Preparation",
    "Cleaning, without leaking the test set",
    "Every step below is fit on the training split only and applied to the test split — never the reverse.",
)

tab1, tab2, tab3 = st.tabs(["Pipeline Steps", "Imputation: Median vs. KNN", "Outliers: IQR vs. 3-Sigma"])

with tab1:
    steps = [
        ("1", "Stratified train/test split — first", """
            Splitting **before** any imputation, scaling, or outlier handling means none of
            those preprocessing statistics ever see the test set. 80/20 split, stratified on
            Outcome so both splits keep the same class ratio.
        """),
        ("2", "Missingness indicator flags", """
            Before imputing, two new binary columns are added: `SkinThickness_was_missing`
            and `Insulin_was_missing`. Plain median-filling would otherwise erase the
            "was this even measured" signal for the two heaviest-missing columns.
        """),
        ("3", "Median imputation (zero → NaN → median)", """
            For Glucose, BloodPressure, SkinThickness, Insulin, and BMI: zeros are treated as
            missing, then filled with the **training-split median** — the same median value
            is then applied to the test split, never recomputed on it. Tested against KNN
            imputation — see the next tab for why median was kept.
        """),
        ("4", "Outlier check (IQR rule, post-imputation)", """
            Boxplots and an IQR-based count run *after* imputation and *before* scaling,
            since StandardScaler is sensitive to outliers. Tested against the 3-sigma rule —
            see the third tab for why IQR was kept.
        """),
        ("5", "EDA / correlation — recomputed post-imputation", """
            The correlation heatmap is rebuilt on the cleaned data. Running it earlier (before
            imputation) would measure the relationship between Outcome and the zero-sentinels,
            not the real biology.
        """),
        ("6", "Standard scaling", """
            `StandardScaler` is fit on the training split only (`fit_transform`), then applied
            to the test split with `transform` — using the training mean and standard
            deviation, not the test set's own.
        """),
    ]

    for num, title, body in steps:
        st.markdown(
            f"""
            <div class="phase-card">
                <span style="color:{TEAL}; font-weight:700; font-family:'Source Serif 4',serif; font-size:1.3rem;">{num}</span>
                &nbsp;&nbsp;<b>{title}</b>
                <div style="margin-top:0.4rem; color:#444; font-size:0.92rem; line-height:1.55;">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="callout-amber">
        <b>Decision point — Insulin/SkinThickness outliers:</b> rather than dropping rows
        (which would shrink an already-small ~614-row training set), this project keeps all
        rows and relies on the missingness flags from Step 2 plus standard scaling. Tree-based
        models in this project's lineup (Decision Tree, Random Forest, Gradient Boosting,
        XGBoost) are not sensitive to these outliers regardless.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    imp = results["experiments"]["imputation"]
    st.markdown("##### Does a smarter imputer actually help?")
    st.markdown(
        """
        Median imputation fills every missing value with the same number, regardless of
        context. KNN imputation instead borrows values from the most similar rows — in
        principle more informed. Tested empirically rather than assumed: same train/test
        split, same Decision Tree (depth=3), only the imputation method differs.
        """
    )

    median_m, knn_m = imp["median"], imp["knn"]
    fig = go.Figure()
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    fig.add_trace(go.Bar(name="Median", x=labels, y=[median_m[m] for m in metrics], marker_color=TEAL))
    fig.add_trace(go.Bar(name="KNN (k=5)", x=labels, y=[knn_m[m] for m in metrics], marker_color=AMBER))
    fig.update_layout(
        barmode="group", height=340, margin=dict(t=10, b=10, l=10, r=10),
        yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"""
        <div class="callout-teal">
        <b>Verdict: median imputation kept.</b> {imp['note']}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Median: Recall={median_m['recall']:.3f}, Precision={median_m['precision']:.3f} · "
        f"KNN: Recall={knn_m['recall']:.3f}, Precision={knn_m['precision']:.3f}"
    )

with tab3:
    out = results["experiments"]["outliers"]
    st.markdown("##### Does the outlier-detection method matter?")
    st.markdown(
        """
        3-sigma flags anything more than 3 standard deviations from the mean — but the mean
        and std it relies on are themselves pulled by extreme values, a problem on skewed
        columns. IQR's quartile basis doesn't have that same feedback loop. Compared below
        on the actual training data, not assumed from theory alone.
        """
    )

    out_df = pd.DataFrame(out["comparison"]).sort_values("skew", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="IQR", x=out_df["column"], y=out_df["iqr_outliers"], marker_color=CORAL))
    fig.add_trace(go.Bar(name="3-Sigma", x=out_df["column"], y=out_df["sigma3_outliers"], marker_color=AMBER))
    fig.update_layout(
        barmode="group", height=360, margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="Outliers flagged (training split)",
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Columns ordered left-to-right by skew (low → high). Sorted ascending so the divergence at the high-skew end is visible.")

    st.markdown(
        f"""
        <div class="callout-teal">
        <b>Verdict: IQR kept as primary method.</b> {out['note']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_skew = out_df.sort_values("skew", ascending=False).head(2)
    st.caption(
        "Most skewed columns: " +
        ", ".join(f"{r.column} (skew={r.skew:.2f}, IQR={r.iqr_outliers}, 3σ={r.sigma3_outliers})" for r in top_skew.itertuples())
    )

footer()
