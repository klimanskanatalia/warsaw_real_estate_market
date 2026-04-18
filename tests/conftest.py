"""Pytest configuration and shared fixtures."""

from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent
