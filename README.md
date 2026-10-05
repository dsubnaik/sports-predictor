# Sports Predictor

Sports Predictor is a Python sports-analytics project for reproducible,
time-aware player-performance modeling. It contains MLB pitcher strikeout work
and an NFL QB passing-yards research pipeline.

## Overview

The project emphasizes end-to-end modeling: data collection, validation,
leakage-safe feature construction, chronological evaluation, baseline
comparison, and automated testing.

## Current Focus: MLB Pitcher Strikeouts

The MLB model predicts pitcher strikeouts per start from historical pitch-level
and game data. It uses rolling pregame features such as recent strikeout
performance, swinging-strike rate, velocity, spin rate, and workload.

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Rolling-average baseline | 2.01 | 2.49 | 0.09 |
| Linear Regression | 1.99 | 2.42 | 0.14 |

## NFL QB Passing-Yards Predictor

The NFL QB predictor uses point-in-time QB form, passing-volume,
history-availability, opponent-defense, and defensive-rank features. Every
feature uses only information available before the target game.

**Current status:** the frozen production candidate is
`sklearn.ensemble.GradientBoostingRegressor` with the canonical 14-feature
contract. It was selected using 2025 validation only; it has not been saved,
deployed, or evaluated on the locked 2026 holdout.

| Model | 2025 MAE | 2025 RMSE | 2025 R² | Status |
| --- | ---: | ---: | ---: | --- |
| Historical average | 73.76 | 96.60 | 0.1130 | Baseline |
| Linear Regression | 66.00 | 82.00 | 0.3609 | Benchmark |
| Random Forest | 65.57 | 83.57 | 0.3361 | Benchmark |
| XGBoost | 64.6208 | 82.0804 | 0.3596 | Benchmark |
| sklearn Gradient Boosting | 64.2678 | 81.6601 | 0.3662 | Frozen candidate |

The selected model's 2025 validation MAE/RMSE/R² are **64.2678 / 81.6601 /
0.3662**. MAE is an average validation error, not an error bound for an
individual prediction. Sportsbook profitability has not been tested.

The 14 feature families cover QB season/recent passing form, QB
season/recent passing-volume form, QB history availability, opponent-defense
passing yards and attempts allowed, defense history availability, and defensive
matchup rank.

The 2026 holdout remains unopened and unscored. Detailed methodology,
experiment history, selection rationale, and limitations are in
[the NFL QB passing-yards technical record](docs/nfl-qb-passing-yards.md).

Next: train the frozen specification on completed 2021–2025 data, persist the
model and metadata, build weekly inference, integrate projections into
Streamlit, and only later evaluate the locked 2026 holdout once.

## Data Pipeline

The MLB pipeline uses Statcast and MLB game data to identify starters,
aggregate pitcher starts, and build leakage-safe rolling features. NFL QB work
uses public nflverse data through the repository's existing loaders and
normalized schedule context.

## Evaluation

Models are evaluated with MAE, RMSE, and R² on chronological partitions rather
than random splits, representing training on completed games and predicting
future games.

## Setup and Commands

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install pandas numpy scikit-learn pybaseball streamlit pytest joblib

python -m pytest
python -m pytest baseball/tests/test_split_data.py
streamlit run app.py
```

Run individual pipeline scripts only after confirming required local data, for
example `python baseball/training/evaluate.py`.

## Testing and Tech Stack

Pytest coverage includes dataset construction, chronological splitting,
evaluation, and NFL QB leakage boundaries. Ordinary tests avoid live network
calls. The stack includes Python, Pandas, NumPy, scikit-learn, XGBoost,
pybaseball, nflverse-backed loaders, Streamlit, pytest, Git, and GitHub.

## Project Structure

```text
sports-predictor/
├── baseball/                 # MLB pitcher strikeout package
├── docs/
│   └── nfl-qb-passing-yards.md
├── football/                 # NFL QB passing-yards research and modeling
├── app.py
└── README.md
```

## Roadmap

### MLB

- [x] Build pitcher-start dataset and rolling features
- [x] Establish baseline and Linear Regression benchmarks
- [ ] Add opponent features, additional models, and projections
- [ ] Integrate MLB projections into the application

### NFL QB Passing Yards

- [x] Freeze the validation-selected model specification
- [ ] Train on completed 2021–2025 data and persist model metadata
- [ ] Build weekly inference and Streamlit projections
- [ ] Perform one-time locked-2026 holdout evaluation
- [ ] Compare projections with sportsbook lines and backtest later
- [ ] Extend to RB rushing-yards modeling

### Future Expansion

- [ ] Expand MLB prediction markets
- [ ] Add other NFL and NBA player-performance models
- [ ] Build a unified prediction interface

## Project Goals

Sports Predictor is an end-to-end, multi-sport machine-learning project focused
on reproducible pipelines, leakage-safe feature engineering, time-aware
evaluation, meaningful baselines, automated testing, and maintainable project
structure.
