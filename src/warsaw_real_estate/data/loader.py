"""Functions for loading raw and processed datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_raw_districts() -> pd.DataFrame:
    """Load the raw districts reference file."""
    return pd.read_excel(RAW_DIR / "districts.xlsx")


def load_final_dataset() -> pd.DataFrame:
    """Load the fully imputed, model-ready dataset."""
    return pd.read_excel(PROCESSED_DIR / "final_dataset.xlsx")
