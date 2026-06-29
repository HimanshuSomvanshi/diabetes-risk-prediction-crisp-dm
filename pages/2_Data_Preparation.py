import streamlit as st

from utils.common import inject_global_css, page_header, footer, load_results, TEAL

st.set_page_config(page_title="Data Preparation · Diabetes CRISP-DM", page_icon="🛠️", layout="wide")
inject_global_css()
results = load_results()

page_header(
    "Phase 3 · Data Preparation",
    "Cleaning, without leaking the test set",
    "Every step below is fit on the training split only and applied to the test split — never the reverse.",
)

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
        is then applied to the test split, never recomputed on it.
    """),
    ("4", "Outlier check (IQR rule, post-imputation)", """
        Boxplots and an IQR-based count run *after* imputation and *before* scaling,
        since StandardScaler is sensitive to outliers. Insulin and SkinThickness — the
        heavily-imputed columns — show the most flagged points, as expected.
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
    models in this project's lineup (Decision Tree, Random Forest, Gradient Boosting) are
    not sensitive to these outliers regardless.
    </div>
    """,
    unsafe_allow_html=True,
)

footer()
