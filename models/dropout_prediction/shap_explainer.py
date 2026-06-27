"""
SHAP explainability layer for the dropout prediction model.
Produces per-child feature importance so coordinators know WHY a child is flagged.
"""

import numpy as np
import pandas as pd
import shap
from loguru import logger


def compute_shap_values(model, X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            # Binary classification — take class 1
            shap_values = shap_values[1]
        df = pd.DataFrame(shap_values, columns=feature_names)
        df["base_value"] = explainer.expected_value if not isinstance(
            explainer.expected_value, list
        ) else explainer.expected_value[1]
        logger.info(f"SHAP values computed for {len(df)} samples")
        return df
    except Exception as exc:
        logger.warning(f"SHAP computation failed: {exc}")
        return pd.DataFrame()


def explain_child(model, X_single: pd.DataFrame, feature_names: list, top_k: int = 5) -> list[dict]:
    """Return top-k contributing factors for a single child prediction."""
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_single)
        if isinstance(sv, list):
            sv = sv[1]
        sv_arr = sv[0] if sv.ndim > 1 else sv
        pairs = sorted(
            zip(feature_names, sv_arr),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_k]
        return [
            {
                "factor": feat,
                "shap_value": round(float(val), 4),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
                "value": float(X_single[feat].iloc[0]) if feat in X_single.columns else None,
            }
            for feat, val in pairs
        ]
    except Exception as exc:
        logger.warning(f"SHAP explanation failed: {exc}")
        return []


FACTOR_LABELS = {
    "attendance_rate": "School Attendance Rate",
    "poverty_rate_lga": "LGA Poverty Rate",
    "conflict_score_lga": "Conflict Intensity in Area",
    "distance_to_school_km": "Distance to Nearest School",
    "disability": "Child has Disability",
    "math_score": "Mathematics Score",
    "literacy_score": "Literacy Score",
    "household_income_usd_day": "Daily Household Income",
    "parent_edu_level": "Parental Education Level",
    "gender_encoded": "Gender (Female = higher risk in some zones)",
    "conflict_displaced": "Conflict-Displaced Household",
    "school_fee_burden": "School Fee Burden",
    "sibling_count": "Number of Siblings",
    "has_birth_certificate": "Has Birth Certificate",
    "meal_programme_access": "Access to School Feeding Programme",
    "academic_composite": "Overall Academic Performance",
    "poverty_distance_interaction": "Poverty × Distance to School",
    "conflict_gender_interaction": "Conflict × Gender Interaction",
}


def humanise_factors(factors: list[dict]) -> list[dict]:
    for f in factors:
        f["label"] = FACTOR_LABELS.get(f["factor"], f["factor"].replace("_", " ").title())
    return factors
