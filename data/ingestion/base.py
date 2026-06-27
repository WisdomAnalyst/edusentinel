"""Base ingestion interface all data-source connectors implement."""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from loguru import logger


class BaseIngester(ABC):
    source_name: str = "unknown"

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Download or load raw data and return a DataFrame."""

    def run(self) -> pd.DataFrame:
        logger.info(f"[{self.source_name}] Starting ingestion …")
        df = self.fetch()
        out = self.raw_dir / f"{self.source_name}.csv"
        df.to_csv(out, index=False)
        logger.success(f"[{self.source_name}] Saved {len(df)} rows → {out}")
        return df
