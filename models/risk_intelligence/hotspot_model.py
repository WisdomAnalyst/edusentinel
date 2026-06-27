"""
Out-of-School Children Risk Intelligence Model
Clusters LGAs into risk hotspots and predicts next-term OOS escalation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib
from loguru import logger

MODEL_DIR = Path(__file__).parent / "artifacts"
RISK_FEATURES = [
    "poverty_rate", "conflict_score", "gender_gap",
    "distance_to_school_km", "disability_rate",
    "school_density_per_1000", "literacy_rate",
    "water_sanitation_access", "teacher_pupil_ratio",
]


def train_hotspot_cluster(lga_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """K-Means clustering of LGAs into risk quintiles."""
    X = lga_df[RISK_FEATURES].copy()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    lga_df = lga_df.copy()
    lga_df["cluster"] = km.fit_predict(X_scaled)

    # Label clusters by mean OOS rate (0 = lowest risk)
    cluster_oos = lga_df.groupby("cluster")["oos_rate"].mean().sort_values()
    rank_map = {v: i for i, v in enumerate(cluster_oos.index)}
    lga_df["risk_cluster"] = lga_df["cluster"].map(rank_map)

    labels = {0: "Low", 1: "Moderate", 2: "Elevated", 3: "High", 4: "Critical"}
    lga_df["cluster_label"] = lga_df["risk_cluster"].map(labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, MODEL_DIR / "kmeans_hotspot.pkl")
    joblib.dump(scaler, MODEL_DIR / "hotspot_scaler.pkl")
    logger.success(f"Hotspot clustering done — {n_clusters} clusters")
    return lga_df


def train_oos_predictor(lga_df: pd.DataFrame) -> GradientBoostingRegressor:
    """Predict next-term OOS rate as a regression target."""
    X = lga_df[RISK_FEATURES]
    # Simulate slight deterioration as "next-term" target
    rng = np.random.default_rng(42)
    y = (lga_df["oos_rate"] * (1 + rng.uniform(-0.05, 0.15, len(lga_df)))).clip(0, 1)

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X, y)
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring="r2").mean()
    logger.success(f"OOS predictor trained — CV R²: {cv_r2:.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "oos_predictor.pkl")
    return model


def load_hotspot_model():
    km = joblib.load(MODEL_DIR / "kmeans_hotspot.pkl")
    scaler = joblib.load(MODEL_DIR / "hotspot_scaler.pkl")
    return km, scaler


def load_oos_predictor():
    return joblib.load(MODEL_DIR / "oos_predictor.pkl")


def predict_next_term(lga_df: pd.DataFrame) -> pd.DataFrame:
    """Add next-term OOS rate predictions to LGA dataframe."""
    model = load_oos_predictor()
    X = lga_df[RISK_FEATURES]
    lga_df = lga_df.copy()
    lga_df["predicted_oos_rate_next_term"] = model.predict(X).clip(0, 1)
    lga_df["predicted_oos_count_next_term"] = (
        lga_df["predicted_oos_rate_next_term"] * lga_df["children_6_14"]
    ).astype(int)
    return lga_df


def run_full_pipeline(lga_df: pd.DataFrame) -> pd.DataFrame:
    lga_df = train_hotspot_cluster(lga_df)
    train_oos_predictor(lga_df)
    lga_df = predict_next_term(lga_df)
    return lga_df
