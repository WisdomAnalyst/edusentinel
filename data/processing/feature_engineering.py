"""
Feature engineering for both the risk intelligence model and the dropout predictor.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger


DROPOUT_FEATURES = [
    "age",
    "gender_encoded",
    "grade_level",
    "attendance_rate",
    "math_score",
    "literacy_score",
    "household_income_usd_day",
    "parent_edu_level",
    "sibling_count",
    "distance_to_school_km",
    "disability",
    "conflict_displaced",
    "school_fee_burden",
    "has_birth_certificate",
    "meal_programme_access",
    "poverty_rate_lga",
    "conflict_score_lga",
    "teacher_pupil_ratio",
]

RISK_FEATURES = [
    "poverty_rate",
    "conflict_score",
    "gender_gap",
    "distance_to_school_km",
    "disability_rate",
    "school_density_per_1000",
    "literacy_rate",
    "water_sanitation_access",
    "teacher_pupil_ratio",
]


def engineer_dropout_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le = LabelEncoder()
    df["gender_encoded"] = le.fit_transform(df["gender"].astype(str))

    # Interaction features
    df["poverty_distance_interaction"] = df["poverty_rate_lga"] * df["distance_to_school_km"]
    df["conflict_gender_interaction"] = df["conflict_score_lga"] * (df["gender_encoded"] == 0).astype(float)
    df["academic_composite"] = (df["math_score"] + df["literacy_score"]) / 200

    extra = [
        "poverty_distance_interaction",
        "conflict_gender_interaction",
        "academic_composite",
    ]
    available = [f for f in DROPOUT_FEATURES + extra if f in df.columns]
    return df[available + ["dropout"]]


def engineer_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zone_le = LabelEncoder()
    df["zone_encoded"] = zone_le.fit_transform(df["geopolitical_zone"])

    df["infrastructure_score"] = (
        df["school_density_per_1000"] / 3 * 0.5 +
        df["water_sanitation_access"] * 0.3 +
        (1 - df["distance_to_school_km"] / 30) * 0.2
    ).clip(0, 1)

    available = [f for f in RISK_FEATURES + ["zone_encoded", "infrastructure_score"] if f in df.columns]
    return df[available]


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    return X_train_s, X_test_s, scaler
