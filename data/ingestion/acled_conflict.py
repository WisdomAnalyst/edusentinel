"""
ACLED Conflict Data Ingester
Armed Conflict Location & Event Data — Nigeria conflict events.
API docs: https://acleddata.com/acleddatanew/wp-content/uploads/dlm_uploads/2017/10/API-User-Guide.pdf
Requires free ACLED API key at https://acleddata.com/register/
"""

import os
from pathlib import Path
import pandas as pd
import requests
from loguru import logger

from .base import BaseIngester

ACLED_BASE = "https://api.acleddata.com/acled/read"


class ACLEDIngester(BaseIngester):
    source_name = "acled_conflict"

    def __init__(self, raw_dir: Path, api_key: str | None = None, email: str | None = None):
        super().__init__(raw_dir)
        self.api_key = api_key or os.getenv("ACLED_API_KEY", "")
        self.email = email or os.getenv("ACLED_EMAIL", "")

    def fetch(self) -> pd.DataFrame:
        if not (self.api_key and self.email):
            logger.warning("ACLED credentials missing — using synthetic conflict scores")
            return self._synthetic_fallback()

        params = {
            "key": self.api_key,
            "email": self.email,
            "country": "Nigeria",
            "year": "2022|2023|2024",
            "fields": "event_date|event_type|admin1|admin2|latitude|longitude|fatalities",
            "limit": 10000,
        }
        try:
            resp = requests.get(ACLED_BASE, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json().get("data", [])
            df = pd.DataFrame(data)
            df = self._aggregate_by_lga(df)
            return df
        except Exception as exc:
            logger.warning(f"ACLED fetch failed ({exc}), using synthetic fallback")
            return self._synthetic_fallback()

    def _aggregate_by_lga(self, df: pd.DataFrame) -> pd.DataFrame:
        df["event_count"] = 1
        agg = df.groupby(["admin1", "admin2"]).agg(
            conflict_events=("event_count", "sum"),
            fatalities=("fatalities", "sum"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        ).reset_index()
        agg.rename(columns={"admin1": "state", "admin2": "lga_name"}, inplace=True)
        agg["conflict_score"] = (agg["conflict_events"] / agg["conflict_events"].max()).clip(0, 1)
        agg["source"] = "ACLED"
        return agg

    def _synthetic_fallback(self) -> pd.DataFrame:
        synthetic = Path(__file__).parent.parent / "raw" / "nigeria_lga_education_indicators.csv"
        if not synthetic.exists():
            return pd.DataFrame()
        df = pd.read_csv(synthetic)
        return df[["lga_name", "state", "latitude", "longitude", "conflict_score"]].assign(
            conflict_events=lambda x: (x["conflict_score"] * 200).astype(int),
            fatalities=lambda x: (x["conflict_score"] * 40).astype(int),
            source="ACLED_synthetic",
        )
