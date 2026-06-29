import streamlit as st

from utils.common import inject_global_css, page_header, footer, load_results

st.set_page_config(page_title="CRISP-DM Summary · Diabetes CRISP-DM", page_icon="🧭", layout="wide")
inject_global_css()
results = load_results()
tc = results["threshold_comparison"]["cost_weighted"]

page_header(
    "Phase Summary",
    "CRISP-DM — complete project walkthrough",
    "All six phases, in one place, with the decisions made at each step.",
)

phases = [
    ("Phase 1 — Business Understanding", [
        "Scope: university project on the PIMA dataset, no specific deployment context defined.",
        "Success criteria: maximize Recall, since a missed diabetic case is treated as costlier than a false alarm.",
        "Cost assumption: False Negative = 4× costlier than False Positive (stated, not derived).",
        "Primary metric: Recall, with the cost ratio disciplining the actual decision threshold.",
    ]),
    ("Phase 2 — Data Understanding", [
        "Source: Kaggle, PIMA Indians Diabetes Database.",
        "Population: female patients of Pima Indian heritage, age 21+ — a stated generalization limit.",
        "Found: 0-values in 5 columns are missing-value sentinels, not true zeros — severity ranges from <5% (Glucose, BMI) to ~49% (Insulin).",
    ]),
    ("Phase 3 — Data Preparation", [
        "Stratified 80/20 split, performed before any preprocessing — no leakage.",
        "Missingness indicator flags added for SkinThickness/Insulin before imputing.",
        "Median imputation fit on train only, applied to test.",
        "Outlier check (IQR) run post-imputation, pre-scaling.",
        "Correlation/EDA recomputed after imputation, not before.",
        "StandardScaler fit on train, applied to test.",
    ]),
    ("Phase 4 — Modeling", [
        "5 classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM.",
        "Each tuned via GridSearchCV, validated with 5-fold Stratified CV — not left at defaults.",
        "Soft-voting ensemble of top 3 by Recall, built for comparison.",
        f"Best individual model by Recall: **{results['best_model_name']}**.",
    ]),
    ("Phase 5 — Evaluation", [
        f"Cost-weighted threshold (minimizing 4×FN + FP) computed: t = {tc['threshold']:.3f}.",
        f"At that threshold: Recall = {tc['recall']:.3f}, Precision = {tc['precision']:.3f}.",
        "Judged acceptable under the stated cost assumption — lower total weighted cost than default or F1-optimized thresholds.",
        "Ship decision: yes, with conditions (population limit, precision tradeoff disclosed, Insulin imputation caveat).",
    ]),
    ("Phase 6 — Deployment", [
        f"Final model ({results['best_model_name']}), scaler, and feature list persisted as joblib artifacts.",
        "Live Predictor page in this app: same preprocessing pipeline as training (missingness flags → imputation → scaling), cost-weighted threshold applied by default.",
        "Predictor shows probability and a precision/recall caveat alongside the label — not a bare binary output.",
    ]),
]

for title, points in phases:
    with st.expander(title, expanded=(title.startswith("Phase 1"))):
        for p in points:
            st.markdown(f"- {p}")

footer()
