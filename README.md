# Sports Predictor

A machine learning and sports analytics project designed to predict player performance using historical and game-level data.

The project currently focuses on **MLB pitcher strikeout prediction**, with plans to expand into additional MLB predictions and eventually support NFL and NBA player performance models.

## Overview

The goal of Sports Predictor is to build an end-to-end machine learning pipeline that collects sports data, engineers predictive features, trains and evaluates models, and generates player performance predictions.

The MLB pitcher strikeout model uses historical pitching data and recent performance trends to estimate the number of strikeouts a starting pitcher will record in an upcoming game.

Rather than focusing only on model training, the project is structured around the full machine learning workflow, including:

* Data collection
* Data validation and preprocessing
* Feature engineering
* Time-based train/validation/test splitting
* Baseline modeling
* Machine learning model training
* Model evaluation
* Automated testing

## Current Focus: MLB Pitcher Strikeouts

The current model predicts:

**Target:** Pitcher strikeouts per start

Historical pitch-level data is aggregated into individual pitcher starts before features are generated for model training.

### Current Features

The model currently incorporates features such as:

* Recent strikeout performance
* Swinging-strike rate
* Average fastball/pitch velocity
* Average spin rate
* Pitch count and workload
* Rolling performance statistics

Rolling features are calculated using only information available **before the game being predicted** to prevent data leakage.

Additional contextual and opponent-based features are planned as the model develops.

## Data Pipeline

The MLB pipeline collects and processes data from multiple sources.

**Statcast data** is used for pitch-level information, including:

* Pitch velocity
* Spin rate
* Swinging strikes
* Pitch counts
* Strikeouts

**MLB game data** is used to identify official starting pitchers and ensure the training dataset contains legitimate pitcher starts.

The resulting pitch-level data is aggregated into a pitcher-start dataset used for feature engineering and model training.

## Modeling

The project uses a baseline-first modeling approach.

### Baseline Model

A rolling strikeout average is used as the initial benchmark.

Current performance:

| Metric | Score |
| ------ | ----: |
| MAE    |  2.01 |
| RMSE   |  2.49 |
| R²     |  0.09 |

### Linear Regression

A Linear Regression model was trained using rolling pitcher performance features.

Current performance:

| Metric | Score |
| ------ | ----: |
| MAE    |  1.99 |
| RMSE   |  2.42 |
| R²     |  0.14 |

The Linear Regression model currently improves upon the baseline. Additional models and features are being evaluated as development continues.

## Model Evaluation

Models are evaluated using:

* **MAE (Mean Absolute Error)** — average prediction error in strikeouts
* **RMSE (Root Mean Squared Error)** — penalizes larger prediction errors
* **R²** — measures the amount of variation explained by the model

Because sports data is time-dependent, the dataset is split chronologically rather than using a random train/test split.

This more closely represents the real-world scenario of training on historical games and predicting future games.

## NFL QB Passing-Yards Modeling

The NFL quarterback passing-yards work follows the same baseline-first,
time-aware approach. It is documented here before Linear Regression begins so
future model comparisons use the same population, boundaries, and validation
standard.

### Completed Components

* A leakage-safe, point-in-time QB passing-yards training dataset.
* A deterministic chronological train/validation/test split.
* A historical-average baseline with a training-only cold-start fallback.
* A structured dataset-quality and baseline-validation audit.
* A read-only real-data smoke run using public nflverse data.
* A held-out, outcome-blind test set. Test outcomes are not used while models
  are developed.

### Dataset Configuration

The smoke run used public nflverse football data through the repository's
existing loaders and `nflreadpy` 0.1.5. Source-history seasons were 2020–2026;
the modeling target rows were restricted to 2021–2026. The extra 2020 season
exists only to provide prior-season history for 2021 Week 1 targets.

* Training target period: 2021–2024
* Validation target period: 2025
* Held-out test target period currently available: 2026 Weeks 1–2
* `validation_start=(2025, 1)`
* `test_start=(2026, 1)`

| Season | Raw player-stat rows | Raw schedule rows |
| ---: | ---: | ---: |
| 2020 | 17,602 | 269 |
| 2021 | 18,969 | 285 |
| 2022 | 18,831 | 284 |
| 2023 | 18,643 | 285 |
| 2024 | 18,983 | 285 |
| 2025 | 19,422 | 285 |
| 2026 | 2,162 | 272 |

The run produced 3,974 normalized QB-game rows, 3,774 normalized schedule
rows, and 3,352 completed modeling rows after the target-season restriction.
Target keys were unique. No loader or normalization warnings occurred.

### Split Summary

| Partition | Rows | Unique QBs | Unique games | Season/week range |
| --- | ---: | ---: | ---: | --- |
| Train | 2,615 | 123 | 1,087 | 2021 W1–2024 W18 |
| Validation | 664 | 81 | 272 | 2025 W1–W18 |
| Test | 73 | 42 | 31 | 2026 W1–W2 |

The audit confirmed consistent schemas, unique target keys within partitions,
no cross-partition key overlap, and correct chronological ordering.

### Feature-History Availability

| Partition | QB season missing | QB last-three missing | Defense season missing | Defense last-three missing |
| --- | ---: | ---: | ---: | ---: |
| Train | 188 (7.19%) | 188 (7.19%) | 0 | 0 |
| Validation | 51 (7.68%) | 51 (7.68%) | 0 | 0 |
| Test | 7 (9.59%) | 7 (9.59%) | 0 | 0 |

Zero-QB-history and cold-start counts equal the missing QB season-history
counts. No defense zero-history rows occurred. Invalid historical sample-size
diagnostics were 0. Negative QB historical-average diagnostics were 10 in
train and 0 in validation and test; negative defense historical-average
diagnostics were 0. Negative passing-yard values can occur in legitimate
low-volume plays and are not automatically data defects.

Sample-size summaries:

* Train QB season history: minimum 0, median 5, maximum 17.
* Validation QB season history: minimum 0, median 5, maximum 17.
* Test QB season history: minimum 0, median 1, maximum 17.
* QB last-three history: minimum 0, median 3, maximum 3 for train and
  validation; minimum 0, median 1, maximum 3 for test.
* Defense season-history median: 9 for train and validation, and 17 for test.
* Defense last-three-history median: 3 in every partition.

### Training Target Summary

This summary is training-only; no equivalent test-target summary was produced.

| Statistic | Value (yards) |
| --- | ---: |
| Rows | 2,615 |
| Mean | 196.15 |
| Sample standard deviation (`ddof=1`) | 103.85 |
| Minimum | -4.0 |
| 25th percentile | 135.0 |
| Median | 210.0 |
| 75th percentile | 268.0 |
| Maximum | 525.0 |

### Official Validation Baseline

The baseline predicts a QB's point-in-time pregame season passing-yards
average. For a cold start with missing history, it uses only the arithmetic
mean target from the training partition: **196.15 yards**. It uses no defense
features and no trained coefficients.

| Validation group | Rows | Share | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall validation | 664 | 100.00% | 73.76 | 96.60 | 0.1130 |
| Fallback/cold-start | 51 | 7.68% | 150.72 | 161.10 | -4.2968 |
| Non-fallback | 613 | 92.32% | 67.36 | 89.16 | 0.1538 |

An MAE of 73.76 means an average absolute miss of about 74 passing yards per
QB-game. RMSE is larger because it penalizes large misses more strongly. An
R² of 0.113 means this simple baseline explains about 11.3% of validation
target variation. Lower MAE/RMSE and higher R² are better; a perfect
zero-error result is not a realistic expectation. Future models must use this
same 2025 validation population and these boundaries for comparison, and must
continue to report cold-start performance separately. This is an official
benchmark, not a production-quality predictor.

### Modeling Population

Train contains 2,615 QB rows across 1,087 games, or about 2.4 QB rows per
game. Backups and low-volume passers are intentionally present. That is a
modeling-population consideration, not a structural audit failure. Any future
population experiment must remain leakage-safe and cannot select starters
using target-game attempts.

### 2026 Test-Set Policy

> **2026 is held out for final model evaluation.** During the smoke run, no
> test predictions, target summaries, residuals, metrics, or outcome-derived
> subgroup analyses were inspected. Test structural and feature-availability
> diagnostics are allowed. Do not evaluate the test set while choosing
> preprocessing, features, hyperparameters, populations, or models.

### Results Ledger

| Model | Validation population | MAE | RMSE | R² | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Historical average | All 2025 eligible QB rows | 73.76 | 96.60 | 0.1130 | Official baseline |

### Planned QB Modeling Stages

1. Training-only preprocessing for missing history.
2. Initial Linear Regression using existing leakage-safe features.
3. Random Forest.
4. Gradient Boosting/XGBoost.
5. Consistent validation comparison and error analysis.
6. Leakage-safe QB-population experiments.
7. Defensive-strength representation experiments.
8. Opponent-adjusted QB form.
9. Defensive-style data investigation using blitz, pressure, man coverage,
   and zone coverage where reliable historical data exists.
10. Defensive-style similarity/clustering experiments.
11. Final model selection.
12. One-time held-out 2026 test evaluation.
13. Model persistence, weekly prediction pipeline, Streamlit projections, and
    sportsbook comparison.
14. RB rushing-yards modeling after the QB pipeline is established.

Defensive tiers are not fixed in advance. Continuous values, thirds,
quartiles, top/bottom groups, or data-derived clusters may be compared using
chronological validation.

## Testing

Automated tests are included to validate important parts of the data and modeling pipeline.

Current tests cover areas such as:

* Statcast data processing
* MLB starting pitcher identification
* Pitcher dataset construction
* Time-based dataset splitting
* Model evaluation

The test suite is run using `pytest`.

## Tech Stack

**Language**

* Python

**Data & Machine Learning**

* Pandas
* NumPy
* scikit-learn
* pybaseball

**Testing & Development**

* pytest
* Git
* GitHub

## Project Structure

```text
sports-predictor/
|
|-- baseball/           # MLB pitcher strikeout prediction package
|   |-- config.py       # Centralized baseball paths
|   |-- data/           # MLB data collection and dataset construction
|   |-- features/       # Feature engineering
|   |-- model/          # Saved-model training and prediction helpers
|   |-- odds/           # MLB odds and prop-line fetching
|   |-- training/       # Training, splitting, and evaluation scripts
|   `-- tests/          # Baseball pipeline tests
|
|-- football/           # NFL research and QB passing-yards modeling package
|-- app.py
`-- README.md
```

The project structure will continue to evolve as additional models and sports are added.

## Roadmap

### MLB

* [x] Build pitcher-start dataset
* [x] Create rolling pitcher features
* [x] Establish baseline model
* [x] Train Linear Regression model
* [x] Add automated pipeline tests
* [ ] Add opponent and matchup features
* [ ] Evaluate additional machine learning models
* [ ] Improve model performance
* [ ] Generate predictions for upcoming games
* [ ] Integrate predictions into the application

### Future Expansion

* [ ] Expand MLB prediction markets
* [ ] Add NFL player performance models
* [ ] Add NBA player performance models
* [ ] Build a unified prediction interface
* [ ] Explore deployment options

## Project Goals

Sports Predictor is being developed as an end-to-end machine learning project focused on applying data science techniques to real-world sports data.

The project emphasizes:

* Reproducible data pipelines
* Leakage-safe feature engineering
* Time-aware model evaluation
* Model comparison against meaningful baselines
* Automated testing
* Maintainable project structure

The long-term goal is to create a multi-sport platform capable of producing data-driven player performance predictions across MLB, NFL, and NBA.
