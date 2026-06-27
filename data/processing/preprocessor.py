"""
Data preprocessor — cleans and merges all ingested data sources
into the master LGA and child-level datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


RAW_DIR = Path(__file__).parent.parent / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "processed"


def load_lga_master() -> pd.DataFrame:
    path = RAW_DIR / "nigeria_lga_education_indicators.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Master LGA file not found at {path}. "
            "Run: python -m data.synthetic.generate_nigeria_data"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded LGA master: {len(df)} rows")
    return df


def load_children_master() -> pd.DataFrame:
    path = RAW_DIR / "nigeria_children_dropout_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Children dataset not found at {path}. "
            "Run: python -m data.synthetic.generate_nigeria_data"
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded children dataset: {len(df)} rows")
    return df


def clean_lga(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        "oos_rate", "poverty_rate", "conflict_score", "gender_gap",
        "distance_to_school_km", "disability_rate", "school_density_per_1000",
        "literacy_rate", "water_sanitation_access", "teacher_pupil_ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    df["oos_count"] = df["oos_count"].fillna(0).astype(int)
    df["risk_score"] = (
        df["oos_rate"] * 0.35 +
        df["poverty_rate"] * 0.25 +
        df["conflict_score"] * 0.20 +
        df["gender_gap"] * 0.10 +
        (df["distance_to_school_km"] / df["distance_to_school_km"].max()) * 0.10
    ).clip(0, 1)
    return df


def clean_children(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attendance_rate"] = df["attendance_rate"].clip(0, 1)
    df["math_score"] = df["math_score"].clip(0, 100)
    df["literacy_score"] = df["literacy_score"].clip(0, 100)
    df["avg_score"] = (df["math_score"] + df["literacy_score"]) / 2
    df["score_percentile"] = df["avg_score"].rank(pct=True)

    cat_cols = ["gender", "state", "geopolitical_zone"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df


def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    lga = clean_lga(load_lga_master())
    children = clean_children(load_children_master())

    lga.to_csv(PROCESSED_DIR / "lga_clean.csv", index=False)
    children.to_csv(PROCESSED_DIR / "children_clean.csv", index=False)
    logger.success("Processing pipeline complete")
    return lga, children


if __name__ == "__main__":
    run_pipeline()
