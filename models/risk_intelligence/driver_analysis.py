"""
Dominant-driver analysis per LGA — identifies which factor (poverty,
conflict, gender, distance, disability, school supply) contributes most
to out-of-school rates and explains them with SHAP-style feature weights.
"""

import pandas as pd
import numpy as np
from loguru import logger


DRIVER_WEIGHTS = {
    "poverty": 0.30,
    "conflict": 0.20,
    "gender": 0.15,
    "distance": 0.15,
    "disability": 0.10,
    "school_supply": 0.10,
}

DRIVER_COLUMNS = {
    "poverty": "poverty_rate",
    "conflict": "conflict_score",
    "gender": "gender_gap",
    "distance": "distance_to_school_km",
    "disability": "disability_rate",
    "school_supply": "school_density_per_1000",
}


def compute_driver_contributions(lga_df: pd.DataFrame) -> pd.DataFrame:
    df = lga_df.copy()

    # Normalise each driver to [0,1] scale
    df["_poverty_n"] = df["poverty_rate"].clip(0, 1)
    df["_conflict_n"] = df["conflict_score"].clip(0, 1)
    df["_gender_n"] = df["gender_gap"].clip(0, 0.6) / 0.6
    df["_distance_n"] = (df["distance_to_school_km"] / 30).clip(0, 1)
    df["_disability_n"] = (df["disability_rate"] * 5).clip(0, 1)
    df["_school_supply_n"] = (1 - (df["school_density_per_1000"] / 3)).clip(0, 1)

    driver_cols = {
        "poverty": "_poverty_n",
        "conflict": "_conflict_n",
        "gender": "_gender_n",
        "distance": "_distance_n",
        "disability": "_disability_n",
        "school_supply": "_school_supply_n",
    }

    for driver, col in driver_cols.items():
        df[f"contribution_{driver}"] = df[col] * DRIVER_WEIGHTS[driver]

    contrib_cols = [f"contribution_{d}" for d in driver_cols]
    df["dominant_driver"] = df[contrib_cols].idxmax(axis=1).str.replace("contribution_", "")
    df["dominant_driver_score"] = df[contrib_cols].max(axis=1)

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)
    return df


def state_level_summary(lga_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate driver contributions at state level."""
    contrib_cols = [f"contribution_{d}" for d in DRIVER_WEIGHTS]
    agg = lga_df.groupby("state").agg(
        oos_rate=("oos_rate", "mean"),
        oos_count=("oos_count", "sum"),
        **{c: (c, "mean") for c in contrib_cols if c in lga_df.columns},
    ).reset_index()

    if contrib_cols:
        available = [c for c in contrib_cols if c in agg.columns]
        if available:
            agg["primary_driver"] = agg[available].idxmax(axis=1).str.replace("contribution_", "")
    return agg


def zone_level_summary(lga_df: pd.DataFrame) -> pd.DataFrame:
    contrib_cols = [f"contribution_{d}" for d in DRIVER_WEIGHTS]
    available = [c for c in contrib_cols if c in lga_df.columns]
    agg_dict = {
        "oos_rate": ("oos_rate", "mean"),
        "oos_count": ("oos_count", "sum"),
    }
    agg_dict.update({c: (c, "mean") for c in available})
    agg = lga_df.groupby("geopolitical_zone").agg(**agg_dict).reset_index()
    if available:
        agg["primary_driver"] = agg[available].idxmax(axis=1).str.replace("contribution_", "")
    return agg
