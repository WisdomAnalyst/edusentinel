"""
GRID3 Nigeria Geospatial Ingester
Geo-Referenced Infrastructure and Demographic Data for Development
Data: https://grid3.gov.ng/
Provides school locations, health facilities, and settlement boundaries.
"""

from pathlib import Path
import pandas as pd
import requests
from loguru import logger

from .base import BaseIngester

GRID3_API = "https://grid3.gov.ng/api/v1"


class GRID3Ingester(BaseIngester):
    source_name = "grid3_geospatial"

    def fetch(self) -> pd.DataFrame:
        cached = Path(__file__).parent.parent / "raw" / "grid3_schools.csv"
        if cached.exists():
            logger.info(f"Loading cached GRID3 school data from {cached}")
            return pd.read_csv(cached)

        try:
            url = f"{GRID3_API}/schools/nigeria"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.DataFrame(resp.json())
            return self._normalise(df)
        except Exception as exc:
            logger.warning(f"GRID3 API unavailable ({exc}), using synthetic fallback")
            return self._synthetic_fallback()

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df["source"] = "GRID3"
        return df

    def _synthetic_fallback(self) -> pd.DataFrame:
        synthetic = Path(__file__).parent.parent / "raw" / "nigeria_lga_education_indicators.csv"
        if not synthetic.exists():
            return pd.DataFrame()
        df = pd.read_csv(synthetic)
        return df[[
            "lga_name", "state", "latitude", "longitude",
            "school_density_per_1000", "teacher_pupil_ratio",
        ]].assign(source="GRID3_synthetic")
