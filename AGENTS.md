# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python sports analytics project focused on MLB pitcher strikeout prediction. The Streamlit entry point is `app.py`. Baseball-specific code lives under `baseball/`: `data/` fetches and builds datasets, `features/` creates leakage-safe rolling features, `training/` contains split/evaluation/model experiment scripts, `model/` handles saved-model training and prediction, `odds/` fetches prop lines, and `tests/` contains pytest coverage. `baseball/config.py` centralizes filesystem paths. `football/` is a placeholder for future work. Generated datasets under `**/data/processed/`, trained models under `**/models/`, `.env`, caches, and `venv/` should stay uncommitted.

## Build, Test, and Development Commands

Create and activate a virtual environment before installing dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn pybaseball streamlit pytest joblib
```

Run the full test suite with `pytest`. Run a focused test with `pytest baseball/tests/test_split_data.py`. Start the local app with `streamlit run app.py`. Run individual pipeline scripts directly, for example `python baseball/training/evaluate.py`, after confirming required data files exist.

## Coding Style & Naming Conventions

Use Python 3 style with 4-space indentation, clear function names, and small modules organized by pipeline stage. Follow existing naming patterns: snake_case for functions, variables, files, and tests; uppercase constants for shared paths in `baseball/config.py`. Prefer `pathlib.Path` for filesystem paths and reuse centralized config values instead of hard-coded paths. Keep feature code time-aware: rolling or historical features must use only data available before the predicted game.

## Testing Guidelines

Tests use `pytest` and live in `baseball/tests/`. Name files `test_<behavior>.py` and test functions `test_<expected_behavior>`. Add focused tests for data transformations, feature engineering, chronological splitting, model evaluation, and API parsing boundaries. Avoid tests that require live network calls unless they are explicitly isolated or mocked.

## Commit & Pull Request Guidelines

Recent commits use short imperative messages such as `Add centralized baseball file paths`, `Update gitignore for generated files`, and `Reorganize baseball paths`. Keep commits scoped to one logical change. Pull requests should describe the model or pipeline behavior changed, list test commands run, note any new data/model artifacts, and include screenshots when the Streamlit UI changes.

## Security & Configuration Tips

Store API keys and local secrets in `.env`; do not commit them. Treat sportsbook lines, Statcast pulls, processed CSVs, and trained model files as reproducible or local artifacts unless the project explicitly decides to version a specific artifact.
