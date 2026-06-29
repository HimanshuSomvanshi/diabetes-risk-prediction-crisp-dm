import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.common import (
    inject_global_css, page_header, metric_card, footer,
    load_results, load_raw_data, TEAL, AMBER, CORAL, SLATE, MUTED, BORDER,
)

st.set_page_config(page_title="Data Understanding · Diabetes CRISP-DM", page_icon="🔍", layout="wide")
inject_global_css()
results = load_results()
df = load_raw_data()

page_header(
    "Phase 2 · Data Understanding",
    "What's actually in the data",
    "Source, population, feature definitions, and — most importantly — its data quality issues.",
)

tab1, tab2, tab3 = st.tabs(["Dataset Profile", "Missing Values", "Correlations"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("##### Source & population")
        st.markdown(
            """
            Obtained from **Kaggle** (PIMA Indians Diabetes Database), originally from the
            National Institute of Diabetes and Digestive and Kidney Diseases.

            Restricted to **female patients of Pima Indian heritage, age 21+**.
            Performance numbers on this page and the Live Predictor describe this
            population only — no claim is made about generalization elsewhere.
            """
        )
        st.markdown("##### Feature definitions")
        feature_defs = pd.DataFrame([
            ("Pregnancies", "Number of times pregnant"),
            ("Glucose", "Plasma glucose concentration (2hr oral glucose tolerance test)"),
            ("BloodPressure", "Diastolic blood pressure (mm Hg)"),
            ("SkinThickness", "Triceps skin fold thickness (mm)"),
            ("Insulin", "2-Hour serum insulin (mu U/ml)"),
            ("BMI", "Body mass index, weight in kg/(height in m)²"),
            ("DiabetesPedigreeFunction", "Function scoring likelihood of diabetes based on family history"),
            ("Age", "Age in years"),
            ("Outcome", "1 = diabetic, 0 = not diabetic (target)"),
        ], columns=["Feature", "Definition"])
        st.dataframe(feature_defs, hide_index=True, width="stretch")

    with col2:
        st.markdown("##### Class balance")
        outcome_counts = results["dataset"]["outcome_counts"]
        fig = go.Figure(data=[go.Pie(
            labels=["Not Diabetic", "Diabetic"],
            values=[outcome_counts["0"], outcome_counts["1"]],
            hole=0.55,
            marker=dict(colors=[TEAL, CORAL]),
            textinfo="label+percent",
        )])
        fig.update_layout(
            showlegend=False, height=300, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter"),
        )
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f"""
            <div class="callout-teal">
            ~{outcome_counts['1']/(outcome_counts['0']+outcome_counts['1'])*100:.0f}% prevalence —
            moderate imbalance. A model that predicts "not diabetic" for everyone would
            already score ~{outcome_counts['0']/(outcome_counts['0']+outcome_counts['1'])*100:.0f}%
            accuracy while being clinically useless. This is why Recall, not Accuracy,
            is this project's primary metric (Phase 1).
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab2:
    st.markdown("##### Zero-as-missing diagnosis")
    st.markdown(
        """
        A value of **0** for Glucose, BloodPressure, SkinThickness, Insulin, or BMI is not
        physiologically possible — it's a missing-value sentinel, not a true measurement.
        The severity differs sharply by column, which matters for how each should be treated.
        """
    )
    zero_counts = results["dataset"]["zero_counts"]
    zc_df = pd.DataFrame([
        {"Column": k, "Zero Count": v["count"], "% of rows": v["pct"]}
        for k, v in zero_counts.items()
    ]).sort_values("% of rows", ascending=False)

    fig = go.Figure(data=[go.Bar(
        x=zc_df["% of rows"], y=zc_df["Column"], orientation="h",
        marker_color=[CORAL if p > 25 else (AMBER if p > 3 else TEAL) for p in zc_df["% of rows"]],
        text=[f"{p:.1f}%" for p in zc_df["% of rows"]], textposition="outside",
    )])
    fig.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=40),
        xaxis_title="% of training rows with value = 0",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        <div class="callout-coral">
        <b>SkinThickness (~30%) and Insulin (~49%)</b> are not "fill a few gaps" — nearly
        half of Insulin is fabricated by a single statistic if median-imputed blindly.
        This project adds a missingness indicator flag for both <i>before</i> imputing,
        preserving the "was this measured at all" signal that plain median-filling would
        destroy (see Phase 3 — Data Preparation).
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab3:
    st.markdown("##### Correlation with Outcome (raw data, before imputation)")
    corr = results["dataset"]["correlations_with_outcome"]
    corr_df = pd.DataFrame(
        [(k, v) for k, v in corr.items() if k != "Outcome"],
        columns=["Feature", "Correlation"],
    ).sort_values("Correlation", ascending=True)

    fig = go.Figure(data=[go.Bar(
        x=corr_df["Correlation"], y=corr_df["Feature"], orientation="h",
        marker_color=TEAL,
    )])
    fig.update_layout(
        height=380, margin=dict(t=10, b=10, l=10, r=20),
        xaxis_title="Correlation with Outcome",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Glucose shows the strongest relationship with Outcome — consistent with "
        "published PIMA benchmarks and with the feature importances in Phase 5."
    )

footer()
