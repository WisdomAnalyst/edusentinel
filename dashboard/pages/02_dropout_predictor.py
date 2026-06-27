"""
Page 2 — School Dropout Early Warning System
Individual risk scoring with SHAP explanations for field coordinators.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Dropout Early Warning | EduSentinel", layout="wide", page_icon="⚠️")

st.title("⚠️ School Dropout Early Warning System")
st.markdown(
    "Score individual children for dropout risk. "
    "Every prediction comes with SHAP-powered explanations so field coordinators "
    "know exactly why a child is flagged and what intervention to deploy."
)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    try:
        from models.dropout_prediction.predictor import load_models, prepare_features
        from models.dropout_prediction.shap_explainer import explain_child, humanise_factors
        xgb_m, lgb_m, features = load_models()
        return xgb_m, lgb_m, features, prepare_features, explain_child, humanise_factors, True
    except Exception as e:
        return None, None, None, None, None, None, False

xgb_m, lgb_m, features, prepare_features, explain_child, humanise_factors, models_loaded = get_models()

if not models_loaded:
    st.warning(
        "Models not yet trained. Run the training pipeline first:\n"
        "```\npython -m models.dropout_prediction.mlflow_pipeline\n```"
    )
    # Show demo mode using pre-set values
    st.info("Running in DEMO mode with sample predictions.")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Individual Risk Scorer", "Batch Analysis", "Model Performance"])

# ─ Tab 1: Individual scorer ───────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Child Profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Demographics**")
        age = st.slider("Age", 6, 15, 10)
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        grade = st.slider("Grade Level", 1, 9, 4)
        sibling_count = st.slider("Number of Siblings", 0, 14, 3)
        has_birth_cert = st.checkbox("Has Birth Certificate", value=True)

    with c2:
        st.markdown("**Academic Profile**")
        attendance = st.slider("Attendance Rate", 0.0, 1.0, 0.72, 0.01, format="%.0%%")
        math_score = st.slider("Mathematics Score", 0, 100, 55)
        literacy_score = st.slider("Literacy Score", 0, 100, 52)
        meal_prog = st.checkbox("School Feeding Programme Access", value=False)
        disability = st.checkbox("Has Disability", value=False)

    with c3:
        st.markdown("**Household & Context**")
        income = st.slider("Daily Household Income (USD)", 0.1, 10.0, 1.5, 0.1)
        parent_edu = st.selectbox(
            "Parental Education",
            [0, 1, 2, 3],
            format_func=lambda x: ["None", "Primary", "Secondary", "Tertiary"][x],
        )
        distance = st.slider("Distance to School (km)", 0.5, 25.0, 3.5, 0.5)
        poverty_lga = st.slider("LGA Poverty Rate", 0.0, 1.0, 0.45, 0.05, format="%.0%%")
        conflict_lga = st.slider("LGA Conflict Score", 0.0, 1.0, 0.20, 0.05)
        conflict_displaced = st.checkbox("Conflict-Displaced Household", value=False)
        fee_burden = st.slider("School Fee Burden", 0.0, 1.0, 0.35, 0.05)

    teacher_ratio = st.slider("Teacher-Pupil Ratio (1:X)", 20, 100, 50)

    if st.button("Predict Dropout Risk", type="primary", use_container_width=True):
        child_input = {
            "age": age,
            "gender": "F" if gender == "Female" else "M",
            "grade_level": grade,
            "attendance_rate": attendance,
            "math_score": float(math_score),
            "literacy_score": float(literacy_score),
            "household_income_usd_day": income,
            "parent_edu_level": parent_edu,
            "sibling_count": sibling_count,
            "distance_to_school_km": distance,
            "disability": int(disability),
            "conflict_displaced": int(conflict_displaced),
            "school_fee_burden": fee_burden,
            "has_birth_certificate": int(has_birth_cert),
            "meal_programme_access": int(meal_prog),
            "poverty_rate_lga": poverty_lga,
            "conflict_score_lga": conflict_lga,
            "teacher_pupil_ratio": 1 / teacher_ratio,
        }

        if models_loaded:
            from models.dropout_prediction.predictor import predict_dropout_risk
            result = predict_dropout_risk(child_input)
            prob = result["dropout_probability"]
            risk = result["risk_level"]
            factors = humanise_factors(result["top_factors"])
        else:
            # Demo fallback
            prob = (
                0.25 * (1 - attendance) +
                0.20 * poverty_lga +
                0.15 * conflict_lga +
                0.10 * (distance / 15) +
                0.10 * int(disability) +
                0.05 * fee_burden +
                (0.15 if gender == "Female" else 0) * poverty_lga +
                np.random.uniform(-0.05, 0.05)
            )
            prob = float(np.clip(prob, 0.02, 0.97))
            risk = (
                "Critical" if prob >= 0.75 else
                "High" if prob >= 0.55 else
                "Moderate" if prob >= 0.35 else "Low"
            )
            factors = [
                {"label": "School Attendance Rate", "shap_value": -0.18 if attendance > 0.7 else 0.22, "direction": "increases_risk" if attendance < 0.7 else "decreases_risk", "value": attendance},
                {"label": "LGA Poverty Rate", "shap_value": poverty_lga * 0.25, "direction": "increases_risk", "value": poverty_lga},
                {"label": "Distance to Nearest School", "shap_value": distance * 0.008, "direction": "increases_risk", "value": distance},
                {"label": "Conflict Intensity in Area", "shap_value": conflict_lga * 0.15, "direction": "increases_risk", "value": conflict_lga},
                {"label": "Overall Academic Performance", "shap_value": -0.12 if (math_score + literacy_score) / 2 > 50 else 0.10, "direction": "increases_risk" if (math_score + literacy_score) / 2 < 50 else "decreases_risk", "value": (math_score + literacy_score) / 200},
            ]

        # ── Result display ────────────────────────────────────────────────────
        RISK_COLORS_UI = {
            "Critical": "#ef4444", "High": "#f97316",
            "Moderate": "#eab308", "Low": "#22c55e",
        }
        risk_color = RISK_COLORS_UI.get(risk, "#6b7280")

        st.markdown("---")
        r1, r2 = st.columns([1, 2])

        with r1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40, "color": risk_color}},
                title={"text": f"Dropout Risk Score<br><b style='color:{risk_color}'>{risk}</b>"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, 35], "color": "#14532d"},
                        {"range": [35, 55], "color": "#713f12"},
                        {"range": [55, 75], "color": "#7c2d12"},
                        {"range": [75, 100], "color": "#450a0a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d2137", font_color="#e8edf5", height=280,
                margin={"t": 40, "b": 10},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with r2:
            st.subheader("Why this child is flagged — SHAP Analysis")
            if factors:
                factor_df = pd.DataFrame(factors)
                factor_df["color"] = factor_df["direction"].map({
                    "increases_risk": "#ef4444",
                    "decreases_risk": "#22c55e",
                })
                fig_shap = go.Figure(go.Bar(
                    x=factor_df["shap_value"],
                    y=factor_df["label"],
                    orientation="h",
                    marker_color=factor_df["color"].tolist(),
                    text=factor_df["shap_value"].map(lambda x: f"{x:+.3f}"),
                    textposition="outside",
                ))
                fig_shap.update_layout(
                    plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
                    font_color="#e8edf5", height=260,
                    xaxis_title="SHAP Value (impact on dropout probability)",
                    margin={"t": 10},
                )
                st.plotly_chart(fig_shap, use_container_width=True)

        # Intervention recommendations
        st.markdown("---")
        st.subheader("Recommended Interventions")
        recs = []
        if child_input["attendance_rate"] < 0.7:
            recs.append("🔴 **Attendance Alert** — Assign home-visit coordinator; investigate absenteeism root cause")
        if child_input["poverty_rate_lga"] > 0.5:
            recs.append("💰 **Conditional Cash Transfer** — Enrol household in CCT programme (NSP/SFTAS)")
        if child_input["distance_to_school_km"] > 5:
            recs.append("🚌 **Transport Support** — Provide school transport subsidy or identify nearest community school")
        if child_input["conflict_displaced"]:
            recs.append("🏕️ **Displaced Child Protocol** — Link to UNICEF emergency education response team")
        if child_input["disability"]:
            recs.append("♿ **Inclusive Education** — Refer to special needs coordinator and adaptive learning programme")
        if gender == "Female" and child_input["poverty_rate_lga"] > 0.4:
            recs.append("👧 **Girl-Child Programme** — Enrol in Kotlead's Gender Equity Access initiative")
        if not recs:
            recs.append("✅ **Monitor** — Low risk, maintain regular attendance checks")

        for r in recs:
            st.markdown(r)

# ─ Tab 2: Batch analysis ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Dropout Risk Analysis")
    st.markdown("Upload a CSV of children records or use the synthetic dataset.")

    col_up, col_sample = st.columns(2)
    with col_up:
        uploaded = st.file_uploader("Upload children CSV", type="csv")
    with col_sample:
        if st.button("Load Synthetic Sample (500 children)"):
            p = Path(__file__).parent.parent.parent / "data" / "raw" / "nigeria_children_dropout_dataset.csv"
            if p.exists():
                st.session_state["batch_df"] = pd.read_csv(p).sample(500, random_state=42)
                st.success("Loaded 500 sample children")

    if uploaded:
        st.session_state["batch_df"] = pd.read_csv(uploaded)

    if "batch_df" in st.session_state:
        batch = st.session_state["batch_df"]
        st.write(f"Loaded {len(batch)} children records")

        if "dropout" in batch.columns:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Children", len(batch))
            col_b.metric("Dropout Rate", f"{batch['dropout'].mean():.1%}")
            col_c.metric("At-Risk (≥35%)", int((batch["dropout"] == 1).sum()))

            fig_zone = px.histogram(
                batch, x="geopolitical_zone", color="dropout",
                barmode="group",
                color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                labels={"geopolitical_zone": "Zone", "count": "Children", "dropout": "Dropout"},
            )
            fig_zone.update_layout(
                plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
                font_color="#e8edf5", height=350,
            )
            st.plotly_chart(fig_zone, use_container_width=True)

            # Feature correlations with dropout
            numeric_cols = batch.select_dtypes(include=[np.number]).columns.tolist()
            if "dropout" in numeric_cols:
                corr = batch[numeric_cols].corr()["dropout"].drop("dropout").sort_values()
                fig_corr = px.bar(
                    x=corr.values,
                    y=corr.index,
                    orientation="h",
                    color=corr.values,
                    color_continuous_scale=["#22c55e", "#94a3b8", "#ef4444"],
                    labels={"x": "Correlation with Dropout", "y": "Feature"},
                    title="Feature Correlation with Dropout",
                )
                fig_corr.update_layout(
                    plot_bgcolor="#0d2137", paper_bgcolor="#0d2137",
                    font_color="#e8edf5", height=400, coloraxis_showscale=False,
                )
                st.plotly_chart(fig_corr, use_container_width=True)

# ─ Tab 3: Model performance ───────────────────────────────────────────────────
with tab3:
    st.subheader("Model Performance Dashboard")

    # Try to load MLflow best run
    try:
        from models.dropout_prediction.mlflow_pipeline import get_best_run
        best = get_best_run()
    except Exception:
        best = None

    if best:
        m = best["metrics"]
        st.success(f"Best run: `{best['run_id'][:12]}…`")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("ROC-AUC", f"{m.get('test_roc_auc', 0):.4f}")
        mc2.metric("F1 Score", f"{m.get('test_f1', 0):.4f}")
        mc3.metric("Precision", f"{m.get('test_precision', 0):.4f}")
        mc4.metric("Recall", f"{m.get('test_recall', 0):.4f}")
    else:
        # Demo metrics
        st.info("No trained model found. Showing demo metrics.")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("ROC-AUC", "0.8921")
        mc2.metric("F1 Score", "0.7843")
        mc3.metric("Precision", "0.8012")
        mc4.metric("Recall", "0.7681")

    st.markdown("---")
    st.subheader("Retrain Model")
    if st.button("Trigger Retraining Pipeline", type="secondary"):
        with st.spinner("Training XGBoost + LightGBM ensemble …"):
            try:
                from models.dropout_prediction.mlflow_pipeline import retrain
                metrics = retrain()
                st.success(f"Retraining complete! AUC: {metrics.get('test_roc_auc', 'N/A')}")
            except Exception as e:
                st.error(f"Training failed: {e}\nMake sure data is generated first.")
