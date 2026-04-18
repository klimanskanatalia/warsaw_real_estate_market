"""Tests for model prediction utilities."""

import pytest


def test_predict_price_import():
    """Verify the predict_price function is importable."""
    from warsaw_real_estate.models.predict import predict_price  # noqa: F401
