# Warsaw Real Estate Market

End-to-end machine-learning pipeline for analysing and predicting residential property prices in Warsaw, Poland.

---

## Project overview

| Stage | Notebook | Description |
|-------|----------|-------------|
| 1 | `notebooks/01_data_ingestion.ipynb` | Scraping & raw data collection |
| 2 | `notebooks/02_data_preparation.ipynb` | Cleaning, type-casting, feature engineering |
| 3 | `notebooks/03_missing_analysis.ipynb` | Missingness patterns & MCAR/MAR analysis |
| 4 | `notebooks/04_data_imputation.ipynb` | Full imputation pipeline |
| 4b | `notebooks/04_data_imputation_simplified.ipynb` | Lightweight imputation variant |
| 5 | `notebooks/05_identify_predictors.ipynb` | Feature importance & model selection |

A Streamlit app (`app/estimate_app.py`) exposes the trained model as an interactive price estimator.

---

## Repository structure

```
.
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── app/
│   └── estimate_app.py           # Streamlit price-estimation UI
├── data/
│   ├── raw/                      # Source data (gitignored – use DVC/cloud)
│   └── processed/                # Cleaned, imputed datasets (gitignored)
├── models/                       # Serialised model artefacts (gitignored)
├── notebooks/                    # Numbered, reproducible Jupyter notebooks
├── reports/
│   └── Analysis_Summary_Report.txt
├── src/
│   └── warsaw_real_estate/       # Installable Python package
│       ├── data/
│       │   └── loader.py         # Dataset loading helpers
│       └── models/
│           └── predict.py        # Inference helpers
├── tests/                        # pytest test suite
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Getting started

### 1. Clone & install

```bash
git clone https://github.com/yourusername/warsaw-real-estate.git
cd warsaw-real-estate

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in editable mode with dev extras
pip install -e ".[dev]"
```

### 2. Install pre-commit hooks

```bash
pre-commit install
```

### 3. Run the notebooks

Open the notebooks in order inside `notebooks/`.  
Each notebook reads from `data/` and writes outputs back to `data/processed/` or `models/`.

### 4. Launch the price estimator

```bash
make run-app
# or directly:
streamlit run app/estimate_app.py
```

---

## Development

| Command | Description |
|---------|-------------|
| `make install-dev` | Install all dev dependencies |
| `make lint` | Run ruff linter |
| `make format` | Auto-format with black + ruff |
| `make typecheck` | Run mypy static type checking |
| `make test` | Run pytest with coverage |
| `make run-app` | Start the Streamlit app |
| `make clean` | Remove cache / build artefacts |

---

## Using the Python API

```python
from warsaw_real_estate.data.loader import load_final_dataset
from warsaw_real_estate.models.predict import predict_price

df = load_final_dataset()

price_per_m2 = predict_price({
    "surface": 55,
    "rooms": 3,
    "floor": 4,
    "building_age": 20,
    "dist_to_metro_km": 0.8,
    "dist_to_centrum_km": 3.5,
    # ... other features
})
print(f"Estimated price/m²: {price_per_m2:,.0f} PLN")
```

---

## CI / CD

GitHub Actions runs on every push/PR to `main` and `develop`:
- **Lint**: ruff
- **Format check**: black
- **Type check**: mypy
- **Tests**: pytest with coverage report

---

## License

[MIT](LICENSE)
