import streamlit as st

from utils.common import (
    inject_global_css, page_header, footer,
    load_results, predict_diabetes_risk,
    SLATE, TEAL, AMBER, CORAL, MUTED,
)

st.set_page_config(page_title="Live Predictor · Diabetes CRISP-DM", page_icon="🎯", layout="wide")
inject_global_css()
results = load_results()
threshold = results["threshold_comparison"]["cost_weighted"]["threshold"]
importances = results["feature_importances"]
best_model = results["best_model_name"]

page_header(
    "Phase 6 · Deployment",
    "Live Predictor",
    f"Same preprocessing as training, {best_model} model, cost-weighted threshold (t={threshold:.3f}).",
)

st.markdown(
    """
    <div class="callout-amber">
    This demonstrates Phase 6 (Deployment) for a university project — no real clinical
    use is implied. Enter values and see what the trained model returns, including the
    tradeoffs behind that output.
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([2, 3])

with left:
    st.markdown("##### Patient values")
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", 0, 17, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0, 250, 120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", 0, 140, 70)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
    with c2:
        insulin = st.number_input("Insulin (mu U/ml)", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 28.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.40, step=0.01)
        age = st.number_input("Age", 21, 100, 33)

    st.caption("Tip: leave a field at 0 for Glucose/BloodPressure/SkinThickness/Insulin/BMI to simulate a missing measurement — the app imputes it the same way training data was imputed.")
    predict_clicked = st.button("Predict", type="primary", width="stretch")

with right:
    if predict_clicked:
        raw = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age,
        }
        result = predict_diabetes_risk(raw, threshold)
        prob = result["probability"]
        is_positive = result["prediction"] == 1

        label = "Diabetic" if is_positive else "Not Diabetic"
        label_color = CORAL if is_positive else TEAL

        st.markdown(
            f"""
            <div style="background:white; border:2px solid {label_color}; border-radius:10px; padding:1.4rem 1.6rem;">
                <div style="font-size:0.8rem; color:{MUTED}; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">
                    Prediction
                </div>
                <div style="font-family:'Source Serif 4',serif; font-size:2.3rem; color:{label_color}; font-weight:700; line-height:1.15;">
                    {label}
                </div>
                <div style="font-size:0.95rem; color:{SLATE}; margin-top:0.3rem;">
                    Probability of diabetes: <b>{prob*100:.1f}%</b> &nbsp;·&nbsp;
                    Decision threshold: <b>{threshold:.3f}</b> (cost-weighted, not 0.5)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Signature element: linear risk spectrum, not a circular gauge ---
        st.write("")
        st.markdown("###### Where this falls on the decision spectrum")
        pct = min(max(prob, 0), 1) * 100
        thresh_pct = threshold * 100
        st.markdown(
            f"""
            <div style="position:relative; height:46px; margin: 0.4rem 0 1.6rem 0;">
                <div style="position:absolute; top:18px; left:0; right:0; height:10px;
                            background:linear-gradient(90deg, {TEAL} 0%, {AMBER} 50%, {CORAL} 100%);
                            border-radius:6px;"></div>
                <div style="position:absolute; top:0px; left:{thresh_pct}%; width:2px; height:46px;
                            background:{SLATE};"></div>
                <div style="position:absolute; top:-16px; left:{thresh_pct}%; transform:translateX(-50%);
                            font-size:0.7rem; color:{SLATE}; font-weight:600; white-space:nowrap;">
                    decision line
                </div>
                <div style="position:absolute; top:14px; left:{pct}%; transform:translateX(-50%);
                            width:18px; height:18px; background:{SLATE}; border-radius:50%;
                            border:3px solid white; box-shadow:0 1px 4px rgba(0,0,0,0.3);"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:{MUTED}; margin-top:-1.2rem;">
                <span>0% — low probability</span>
                <span>100% — high probability</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            f"The threshold sits at {threshold:.3f}, not 0.5, because Phase 5 found that "
            f"minimizes total cost under a 4:1 FN:FP assumption — not because it maximizes accuracy."
        )

        # --- Precision/Recall caveat ---
        cw = results["threshold_comparison"]["cost_weighted"]
        st.markdown(
            f"""
            <div class="callout-coral">
            <b>Read this before trusting the label:</b> at this threshold the model catches
            {cw['recall']*100:.0f}% of true diabetic cases, but only {cw['precision']*100:.0f}%
            of positive predictions are actually correct — roughly {cw['fp']} false alarms for
            every {cw['tp']} true positives on the test set. A "Diabetic" result here is a
            screening flag, not a diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --- Feature driver panel, grounded in real importances ---
        st.write("")
        st.markdown("###### What actually drove this prediction")
        nonzero_features = {k: v for k, v in importances.items() if v > 0}
        feature_values = {
            "Glucose": glucose, "BMI": bmi, "Age": age,
        }
        max_imp = max(nonzero_features.values()) if nonzero_features else 1
        for feat, imp in sorted(nonzero_features.items(), key=lambda x: -x[1]):
            val = feature_values.get(feat, raw.get(feat, "—"))
            bar_pct = imp / max_imp * 100
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid #E5E1DA;">
                    <div style="width:90px; font-weight:600; font-size:0.88rem;">{feat}</div>
                    <div style="flex:1; background:#E5E1DA; height:8px; border-radius:4px; overflow:hidden;">
                        <div style="width:{bar_pct:.0f}%; height:100%; background:{TEAL};"></div>
                    </div>
                    <div style="width:60px; text-align:right; font-size:0.82rem; color:{MUTED};">{imp:.1%}</div>
                    <div style="width:70px; text-align:right; font-size:0.85rem;">value: {val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        zero_features = [k for k, v in importances.items() if v == 0]
        st.caption(
            f"This {best_model} (depth=3) only ever splits on Glucose, BMI, and Age — "
            f"the other {len(zero_features)} inputs you entered ({', '.join(zero_features)}) "
            "had zero effect on this specific prediction. See Phase 6 — Feature Analysis "
            "for the full picture, and the note about deploying a different model type "
            "(Random Forest / SVM) if all inputs should contribute."
        )
    else:
        st.markdown(
            f"""
            <div style="background:white; border:1px dashed #C7C2B8; border-radius:10px;
                        padding:2.5rem; text-align:center; color:{MUTED};">
                Fill in the patient values and click <b>Predict</b> to see the result.
            </div>
            """,
            unsafe_allow_html=True,
        )

footer()
