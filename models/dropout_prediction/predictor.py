"""
School Dropout Early Warning System — XGBoost + LightGBM ensemble
with automated MLflow experiment tracking and SHAP explainability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import joblib
import mlflow
import mlflow.sklearn
from loguru import logger

from .shap_explainer import compute_shap_values, explain_child

MODEL_DIR = Path(__file__).parent / "artifacts"

FEATURE_COLS = [
    "age", "gender_encoded", "grade_level", "attendance_rate",
    "math_score", "literacy_score", "household_income_usd_day",
    "parent_edu_level", "sibling_count", "distance_to_school_km",
    "disability", "conflict_displaced", "school_fee_burden",
    "has_birth_certificate", "meal_programme_access",
    "poverty_rate_lga", "conflict_score_lga", "teacher_pupil_ratio",
    "poverty_distance_interaction", "conflict_gender_interaction",
    "academic_composite",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df["gender_encoded"] = le.fit_transform(df["gender"].astype(str))
    df["poverty_distance_interaction"] = df["poverty_rate_lga"] * df["distance_to_school_km"]
    df["conflict_gender_interaction"] = (
        df["conflict_score_lga"] * (df["gender_encoded"] == 0).astype(float)
    )
    df["academic_composite"] = (df["math_score"] + df["literacy_score"]) / 200
    return df


def train_xgboost(X_train, y_train, X_val, y_val) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    return model


def ensemble_predict_proba(xgb_model, lgb_model, X) -> np.ndarray:
    p_xgb = xgb_model.predict_proba(X)[:, 1]
    p_lgb = lgb_model.predict_proba(X)[:, 1]
    return (p_xgb * 0.5 + p_lgb * 0.5)


def run_training_pipeline(
    children_df: pd.DataFrame,
    experiment_name: str = "EduSentinel_Dropout_Prediction",
    threshold: float = 0.45,
) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_features(children_df)
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    X = df[available_features]
    y = df["dropout"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="xgb_lgb_ensemble_v1"):
        mlflow.log_params({
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "dropout_rate_train": round(float(y_train.mean()), 4),
            "threshold": threshold,
            "features": len(available_features),
        })

        logger.info("Training XGBoost …")
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)

        logger.info("Training LightGBM …")
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

        # Ensemble evaluation
        proba = ensemble_predict_proba(xgb_model, lgb_model, X_test)
        preds = (proba >= threshold).astype(int)

        metrics = {
            "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
            "test_f1": round(f1_score(y_test, preds), 4),
            "test_precision": round(precision_score(y_test, preds), 4),
            "test_recall": round(recall_score(y_test, preds), 4),
        }
        mlflow.log_metrics(metrics)

        logger.success(f"Ensemble metrics: {metrics}")
        logger.info(f"\n{classification_report(y_test, preds, target_names=['Enrolled', 'Dropout'])}")

        # Persist models
        joblib.dump(xgb_model, MODEL_DIR / "xgb_dropout.pkl")
        joblib.dump(lgb_model, MODEL_DIR / "lgb_dropout.pkl")
        joblib.dump(available_features, MODEL_DIR / "feature_cols.pkl")
        mlflow.sklearn.log_model(xgb_model, "xgb_model")
        mlflow.sklearn.log_model(lgb_model, "lgb_model")

        # SHAP on test set
        shap_df = compute_shap_values(xgb_model, X_test, available_features)
        shap_df.to_csv(MODEL_DIR / "shap_test.csv", index=False)

        return {**metrics, "features": available_features, "threshold": threshold}


def load_models():
    xgb_model = joblib.load(MODEL_DIR / "xgb_dropout.pkl")
    lgb_model = joblib.load(MODEL_DIR / "lgb_dropout.pkl")
    features = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return xgb_model, lgb_model, features


def predict_dropout_risk(child_data: dict) -> dict:
    """Score a single child record and return risk score + explanation."""
    xgb_model, lgb_model, features = load_models()
    df = pd.DataFrame([child_data])
    df = prepare_features(df)
    X = df[features]

    proba = float(ensemble_predict_proba(xgb_model, lgb_model, X)[0])
    risk_level = (
        "Critical" if proba >= 0.75 else
        "High" if proba >= 0.55 else
        "Moderate" if proba >= 0.35 else "Low"
    )
    explanation = explain_child(xgb_model, X, features)
    return {
        "dropout_probability": round(proba, 4),
        "risk_level": risk_level,
        "top_factors": explanation,
    }
