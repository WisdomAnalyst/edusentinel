"""Dropout prediction API — individual scoring and batch analysis."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import pandas as pd
from loguru import logger

from api.schemas.models import ChildProfile, DropoutPredictionResponse, RetrainResponse

router = APIRouter(prefix="/predictions", tags=["Predictions"])

INTERVENTION_MAP = {
    "Critical": ["conditional_cash_transfer", "home_visit", "emergency_enrolment"],
    "High": ["conditional_cash_transfer", "attendance_monitoring", "school_transport"],
    "Moderate": ["attendance_monitoring", "academic_support"],
    "Low": ["routine_monitoring"],
}

PROGRAMME_NAMES = {
    "conditional_cash_transfer": "National Social Safety Net Programme (NSSNP)",
    "home_visit": "Kotlead Home Visit & Family Engagement Protocol",
    "emergency_enrolment": "UNICEF Emergency Enrolment Drive",
    "attendance_monitoring": "Biometric Attendance Tracking System",
    "school_transport": "Free School Transport Initiative (FG/UBEC)",
    "academic_support": "After-School Remedial Classes Programme",
    "routine_monitoring": "Quarterly Welfare Check-In",
}


@router.post("/score-child", response_model=DropoutPredictionResponse)
def score_child(profile: ChildProfile):
    try:
        from models.dropout_prediction.predictor import predict_dropout_risk
        from models.dropout_prediction.shap_explainer import humanise_factors
        result = predict_dropout_risk(profile.model_dump())
        result["top_factors"] = humanise_factors(result["top_factors"])
    except FileNotFoundError:
        # Demo mode: deterministic heuristic scoring
        result = _heuristic_score(profile)

    risk = result["risk_level"]
    interventions = INTERVENTION_MAP.get(risk, ["routine_monitoring"])
    programmes = [PROGRAMME_NAMES.get(i, i) for i in interventions]

    return DropoutPredictionResponse(
        dropout_probability=result["dropout_probability"],
        risk_level=risk,
        top_factors=result["top_factors"],
        intervention_priority="Immediate" if risk == "Critical" else
                               "Within 2 weeks" if risk == "High" else
                               "This term" if risk == "Moderate" else "Routine",
        recommended_programmes=programmes,
    )


def _heuristic_score(profile: ChildProfile) -> dict:
    import numpy as np
    prob = (
        0.25 * (1 - profile.attendance_rate) +
        0.20 * profile.poverty_rate_lga +
        0.15 * profile.conflict_score_lga +
        0.10 * min(profile.distance_to_school_km / 15, 1.0) +
        0.10 * float(profile.disability) +
        0.08 * profile.school_fee_burden +
        0.07 * float(profile.conflict_displaced) +
        0.05 * (1 - profile.has_birth_certificate)
    )
    if profile.gender == "F":
        prob += profile.poverty_rate_lga * 0.12
    prob = float(np.clip(prob, 0.01, 0.98))
    risk = (
        "Critical" if prob >= 0.75 else
        "High" if prob >= 0.55 else
        "Moderate" if prob >= 0.35 else "Low"
    )
    return {
        "dropout_probability": round(prob, 4),
        "risk_level": risk,
        "top_factors": [
            {"label": "School Attendance Rate", "shap_value": round(-0.15 + (1 - profile.attendance_rate) * 0.3, 4), "direction": "increases_risk" if profile.attendance_rate < 0.8 else "decreases_risk", "value": profile.attendance_rate},
            {"label": "LGA Poverty Rate", "shap_value": round(profile.poverty_rate_lga * 0.2, 4), "direction": "increases_risk", "value": profile.poverty_rate_lga},
            {"label": "Distance to School", "shap_value": round(profile.distance_to_school_km * 0.006, 4), "direction": "increases_risk", "value": profile.distance_to_school_km},
        ],
    }


@router.post("/retrain", response_model=RetrainResponse)
def trigger_retrain(background_tasks: BackgroundTasks):
    def _retrain():
        try:
            from models.dropout_prediction.mlflow_pipeline import retrain
            metrics = retrain()
            logger.success(f"Background retrain complete: {metrics}")
        except Exception as e:
            logger.error(f"Background retrain failed: {e}")

    background_tasks.add_task(_retrain)
    return RetrainResponse(
        status="accepted",
        experiment="EduSentinel_Dropout_Prediction",
        metrics={},
        run_id=None,
    )


@router.get("/model-status")
def get_model_status():
    from pathlib import Path
    model_dir = Path(__file__).parent.parent.parent / "models" / "dropout_prediction" / "artifacts"
    xgb_exists = (model_dir / "xgb_dropout.pkl").exists()
    lgb_exists = (model_dir / "lgb_dropout.pkl").exists()

    if xgb_exists and lgb_exists:
        from models.dropout_prediction.mlflow_pipeline import get_best_run
        best = get_best_run()
        return {
            "status": "ready",
            "models": ["XGBoost", "LightGBM"],
            "best_run": best,
        }
    return {"status": "not_trained", "models": [], "best_run": None}
