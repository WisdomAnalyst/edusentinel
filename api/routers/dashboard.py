"""Dashboard data API — serves LGA risk data, summaries, and driver analysis."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd

from api.schemas.models import LGAFilter

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def _load_lga() -> pd.DataFrame:
    p = DATA_DIR / "nigeria_lga_education_indicators.csv"
    if not p.exists():
        raise HTTPException(
            status_code=503,
            detail="LGA dataset not found. Run the data generation pipeline first.",
        )
    return pd.read_csv(p)


@router.get("/summary")
def get_summary():
    p = DATA_DIR / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    df = _load_lga()
    return {
        "total_lgas": int(len(df)),
        "total_states": int(df["state"].nunique()),
        "total_oos_children": int(df["oos_count"].sum()),
        "national_oos_rate": round(float(df["oos_rate"].mean()), 4),
        "critical_lgas": int((df["risk_tier"] == "Critical").sum()),
        "high_risk_lgas": int((df["risk_tier"] == "High").sum()),
    }


@router.get("/lgas")
def get_lgas(
    zone: str | None = Query(None),
    state: str | None = Query(None),
    risk_tier: str | None = Query(None),
    dominant_driver: str | None = Query(None),
    min_oos_rate: float = Query(0.0, ge=0, le=1),
    limit: int = Query(500, le=774),
):
    df = _load_lga()
    if zone:
        df = df[df["geopolitical_zone"] == zone]
    if state:
        df = df[df["state"] == state]
    if risk_tier:
        df = df[df["risk_tier"] == risk_tier]
    if dominant_driver:
        df = df[df["dominant_driver"] == dominant_driver]
    df = df[df["oos_rate"] >= min_oos_rate]
    return df.head(limit).to_dict(orient="records")


@router.get("/states")
def get_state_summary():
    df = _load_lga()
    agg = df.groupby("state").agg(
        oos_count=("oos_count", "sum"),
        oos_rate=("oos_rate", "mean"),
        poverty_rate=("poverty_rate", "mean"),
        conflict_score=("conflict_score", "mean"),
        total_lgas=("lga_id", "count"),
        critical_lgas=("risk_tier", lambda x: (x == "Critical").sum()),
    ).reset_index().sort_values("oos_rate", ascending=False)
    return agg.to_dict(orient="records")


@router.get("/zones")
def get_zone_summary():
    df = _load_lga()
    agg = df.groupby("geopolitical_zone").agg(
        oos_count=("oos_count", "sum"),
        oos_rate=("oos_rate", "mean"),
        dominant_driver=("dominant_driver", lambda x: x.mode()[0] if len(x) > 0 else "unknown"),
        lga_count=("lga_id", "count"),
    ).reset_index().sort_values("oos_rate", ascending=False)
    return agg.to_dict(orient="records")


@router.get("/hotspots")
def get_top_hotspots(n: int = Query(20, ge=5, le=100)):
    """Top N most critical LGAs ranked by OOS rate."""
    df = _load_lga()
    top = df.nlargest(n, "oos_rate")[[
        "lga_name", "state", "geopolitical_zone", "latitude", "longitude",
        "oos_rate", "oos_count", "risk_tier", "dominant_driver",
        "poverty_rate", "conflict_score", "gender_gap", "distance_to_school_km",
    ]]
    return top.to_dict(orient="records")
