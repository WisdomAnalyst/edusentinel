"""
Nigeria National Bureau of Statistics (NBS) Ingester
Data portal: https://nigerianstat.gov.ng/
Provides poverty indices, household surveys, and enrollment statistics.
"""

from pathlib import Path
import pandas as pd
import requests
from loguru import logger

from .base import BaseIngester

NBS_OPEN_DATA = "https://nigerianstat.gov.ng/elibrary/read"


class NBSIngester(BaseIngester):
    source_name = "nbs_nigeria"

    def fetch(self) -> pd.DataFrame:
        # NBS doesn't have a machine-readable API — we load from cached CSVs
        # or fall back to the synthetic dataset
        cached = Path(__file__).parent.parent / "raw" / "nbs_poverty_2023.csv"
        if cached.exists():
            logger.info(f"Loading cached NBS data from {cached}")
            return pd.read_csv(cached)

        logger.warning("NBS live data not cached — using synthetic poverty indicators")
        return self._synthetic_fallback()

    def _synthetic_fallback(self) -> pd.DataFrame:
        synthetic = Path(__file__).parent.parent / "raw" / "nigeria_lga_education_indicators.csv"
        if not synthetic.exists():
            return pd.DataFrame()
        df = pd.read_csv(synthetic)
        return df[[
            "state", "lga_name", "poverty_rate", "literacy_rate",
            "water_sanitation_access", "geopolitical_zone",
        ]].assign(source="NBS_synthetic", year=2023)
