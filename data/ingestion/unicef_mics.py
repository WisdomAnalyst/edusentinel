"""
UNICEF MICS Nigeria Ingester
Loads Multiple Indicator Cluster Survey education indicators.
Falls back to synthetic data when the live endpoint is unavailable.
Real data: https://mics.unicef.org/surveys
"""

from pathlib import Path
import pandas as pd
import requests
from loguru import logger

from .base import BaseIngester

# UNICEF open-data API for Nigeria MICS summary stats
UNICEF_API = "https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/data"
NGA_EDU_FLOW = "UNICEF,EDUCATION,1.0/NGA.ED_ANAR_L02+ED_ANAR_L1.._T+F+M.?format=csv"


class UNICEFMICSIngester(BaseIngester):
    source_name = "unicef_mics"

    def fetch(self) -> pd.DataFrame:
        # Try live API first
        url = f"{UNICEF_API}/{NGA_EDU_FLOW}"
        try:
            logger.info(f"Fetching UNICEF SDMX: {url}")
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(resp.text))
            df = self._normalise(df)
            return df
        except Exception as exc:
            logger.warning(f"UNICEF API unavailable ({exc}), falling back to synthetic data")
            return self._synthetic_fallback()

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "INDICATOR": "indicator",
            "SEX": "gender",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "value",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["source"] = "UNICEF_MICS"
        return df

    def _synthetic_fallback(self) -> pd.DataFrame:
        """Return minimal MICS-shaped frame from already-generated synthetic data."""
        synthetic = Path(__file__).parent.parent / "raw" / "nigeria_lga_education_indicators.csv"
        if not synthetic.exists():
            logger.warning("Synthetic data not found — run generate_nigeria_data.py first")
            return pd.DataFrame()
        df = pd.read_csv(synthetic)
        mics = pd.DataFrame({
            "indicator": "adjusted_net_attendance_rate_primary",
            "gender": "both",
            "state": df["state"],
            "year": 2023,
            "value": (1 - df["oos_rate"]) * 100,
            "source": "UNICEF_MICS_synthetic",
        })
        return mics
