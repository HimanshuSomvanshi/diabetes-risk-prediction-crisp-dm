"""
Shared styling, constants, and data-loading helpers for the Diabetes CRISP-DM app.
Single source of truth for colors, fonts, and cached data so every page stays consistent.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
SLATE = "#1C2B33"
TEAL = "#2D9596"
TEAL_LIGHT = "#E3F2F2"
AMBER = "#E8A33D"
AMBER_LIGHT = "#FBF0DC"
CORAL = "#D8634C"
CORAL_LIGHT = "#FBE9E5"
CREAM = "#F7F5F2"
WHITE = "#FFFFFF"
INK = "#2A2A2A"
MUTED = "#6B7280"
BORDER = "#E5E1DA"

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def inject_global_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            color: {INK};
        }}

        h1, h2, h3 {{
            font-family: 'Source Serif 4', serif !important;
            color: {SLATE} !important;
            letter-spacing: -0.01em;
        }}

        .stApp {{
            background-color: {CREAM};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {SLATE};
        }}
        section[data-testid="stSidebar"] * {{
            color: {CREAM} !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: rgba(247,245,242,0.15);
        }}

        .eyebrow {{
            font-family: 'Inter', sans-serif;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {TEAL};
            margin-bottom: 0.2rem;
        }}

        .phase-card {{
            background: {WHITE};
            border: 1px solid {BORDER};
            border-left: 4px solid {TEAL};
            border-radius: 6px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem;
        }}

        .metric-card {{
            background: {WHITE};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            text-align: left;
        }}
        .metric-card .label {{
            font-size: 0.78rem;
            color: {MUTED};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
        }}
        .metric-card .value {{
            font-family: 'Source Serif 4', serif;
            font-size: 2.0rem;
            color: {SLATE};
            font-weight: 700;
            line-height: 1.2;
        }}

        .callout-amber {{
            background: {AMBER_LIGHT};
            border: 1px solid {AMBER};
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            font-size: 0.92rem;
        }}
        .callout-teal {{
            background: {TEAL_LIGHT};
            border: 1px solid {TEAL};
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            font-size: 0.92rem;
        }}
        .callout-coral {{
            background: {CORAL_LIGHT};
            border: 1px solid {CORAL};
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            font-size: 0.92rem;
        }}

        .divider-line {{
            border: none;
            border-top: 1px solid {BORDER};
            margin: 1.4rem 0;
        }}

        .footer-note {{
            text-align: center;
            color: {MUTED};
            font-size: 0.78rem;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid {BORDER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<p style="color:{MUTED}; margin-top:-0.4rem;">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)


def metric_card(label: str, value: str, col):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="footer-note">
            HTW Berlin · MSc Data Science &amp; Project Management · CRISP-DM Project ·
            Decision Tree (depth=3) · Cost-weighted threshold (4:1 FN:FP)
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_results() -> dict:
    with open(MODELS_DIR / "results.json") as f:
        return json.load(f)


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "diabetes.csv")


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    columns = joblib.load(MODELS_DIR / "feature_columns.joblib")
    return model, scaler, columns


def predict_diabetes_risk(raw_features: dict, threshold: float) -> dict:
    """Replicates the exact preprocessing used at training time: missingness flags
    derived from raw zeros, then median imputation using the saved train medians,
    then scaling, then prediction at the supplied threshold."""
    model, scaler, columns = load_model_artifacts()
    results = load_results()
    train_medians = results["train_medians"]

    row = {col: raw_features.get(col, 0) for col in [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]}
    df = pd.DataFrame([row])

    for col in ["SkinThickness", "Insulin"]:
        df[f"{col}_was_missing"] = (df[col] == 0).astype(int)

    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if df.loc[0, col] == 0:
            df[col] = train_medians[col]

    df = df[columns]
    scaled = pd.DataFrame(scaler.transform(df), columns=columns)
    prob = float(model.predict_proba(scaled)[0, 1])
    pred = int(prob >= threshold)
    return {"prediction": pred, "probability": prob, "threshold_used": threshold}
