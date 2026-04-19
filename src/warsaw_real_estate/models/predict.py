"""Model loading and inference helpers."""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import pandas as pd



MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def load_feature_columns() -> list[str]:
    """Return the ordered list of feature column names expected by the model."""
    with open(MODELS_DIR / "feature_columns.pkl", "rb") as fh:
        return pickle.load(fh)


def load_model(model_filename: str = "price_model.pkl"):
    """Load and return a trained scikit-learn compatible model."""
    return joblib.load(MODELS_DIR / model_filename)


def predict_price(input_data: dict, model_filename: str = "price_model.pkl") -> float:
    """
    Predict the price per m² for a single property.

    Parameters
    ----------
    input_data:
        Dictionary of feature name → value matching the model's expected features.
    model_filename:
        Filename (relative to the models/ directory) of the persisted model.

    Returns
    -------
    Predicted price per m² in PLN.
    """
    feature_columns = load_feature_columns()
    model = load_model(model_filename)
    df = pd.DataFrame([input_data]).reindex(columns=feature_columns, fill_value=False)
    return float(model.predict(df)[0])
