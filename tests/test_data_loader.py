"""Tests for data loading utilities."""

from pathlib import Path


def test_data_dirs_exist():
    """Ensure the expected data directories are present in the repository."""
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "data" / "raw").is_dir()
    assert (repo_root / "data" / "processed").is_dir()
    assert (repo_root / "models").is_dir()
