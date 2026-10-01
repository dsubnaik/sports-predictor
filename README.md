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
time-aware approach. It is documented here so future model comparisons use the
same population, boundaries, and validation standard.

### Completed Components

* A leakage-safe, point-in-time QB passing-yards training dataset.
* A deterministic chronological train/validation/test split.
* A historical-average baseline with a training-only cold-start fallback.
* A structured dataset-quality and baseline-validation audit.
* A read-only real-data smoke run using public nflverse data.
* Fixed, reproducible Linear Regression, Random Forest, sklearn Gradient
  Boosting, and XGBoost benchmarks.
* A consistent, leakage-safe validation comparison and error-analysis report
  for all five completed models.
* A dated-depth-chart feasibility audit and likely-primary-QB population
  analysis.
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

### Linear Regression Validation

The first model uses `sklearn.linear_model.LinearRegression` with default
parameters. It was trained once on 2021-2024 target rows and evaluated on the
fixed 2025 validation population. The fitted pipeline keeps training-only
preprocessing and the estimator together.

The model uses exactly the public QB passing-yards `FEATURE_COLUMNS` from the
dataset builder. It uses neither identifiers, player names, team or opponent
labels, home/away, sportsbook lines, nor target values as inputs. There is no
feature scaling. Permitted missing numeric history is median-imputed using the
training partition only; the existing boolean missing-history flags are
validated and passed through unchanged.

Included point-in-time feature families are:

* QB history: season and last-three passing-yards averages; season and
  last-three passing-attempt averages; season and last-three history sample
  sizes; and the missing-QB-history flag.
* Opponent defense: season and last-three passing-yards-allowed averages;
  passing attempts allowed; season and last-three defensive history sample
  sizes; the missing-defense-history flag; and point-in-time matchup rank.

Every historical feature contains only information available before its target
game. The model and preprocessing were fitted using 2021-2024 only and stayed
fixed throughout 2025 validation. Thus, a Week 10 row can use QB and defensive
information through Week 9, but the model is not retrained after each 2025
week. This is a fixed-season model with weekly updated point-in-time features;
expanding-window weekly retraining remains a later experiment.

| Group | Rows | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall | 664 | 66.00 | 82.00 | 0.3609 |
| Cold start | 51 | 76.30 | 82.67 | -0.3948 |
| Non-cold-start | 613 | 65.14 | 81.94 | 0.2852 |

Compared with the historical-average baseline on the identical validation
rows, Linear Regression improved MAE by 7.76 yards, RMSE by 14.60 yards, and
R² by 0.2479. Using `baseline - Linear Regression` for MAE and RMSE, and
`Linear Regression - baseline` for R², positive values mean improvement. MAE
decreased by about 10.5% and RMSE by about 15.1%.

An MAE of 66.00 means an average absolute error of about 66 passing yards per
eligible QB-game. The result is a meaningful improvement over the simple
baseline, but remains an experimental benchmark rather than a production
quality betting predictor. It applies to all eligible QB rows, including
backups and low-volume passers; it is not performance restricted to likely
starters or players with sportsbook props. A controlled feature-ablation
experiment is required before attributing this improvement specifically to
opponent defense. Lower MAE/RMSE and higher R² are better.

For reproducibility, the fitted intercept was 98.6586. Selected training-only
median imputations were QB season passing yards 222.33, QB last-three passing
yards 219.67, QB season passing attempts 31.2, QB last-three passing attempts
31.0, defense season passing yards allowed 232.64, defense last-three passing
yards allowed 231.33, and defensive matchup rank 17.0. No numerical-
conditioning warnings occurred. Coefficients are available through the public
model report for reproducibility; they are not feature importance.

### Random Forest Validation

The Random Forest benchmark uses `sklearn.ensemble.RandomForestRegressor` with
one fixed, untuned configuration: `n_estimators=500`, `random_state=42`,
`max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`,
`max_features=1.0`, `bootstrap=True`, and `n_jobs=1`. These parameters were
predetermined, not selected using 2025 validation results.

It uses the exact same 14 public predictive features as Linear Regression:
training-only median imputation for permitted missing numeric history,
validated/pass-through missing-history flags, no scaling, and no row dropping.
It trained on 2021-2024 and remained fixed through 2025 validation while the
point-in-time weekly features updated. Identifiers, target values, sportsbook
lines, home/away, and team labels are not model inputs.

| Group | Rows | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall | 664 | 65.57 | 83.57 | 0.3361 |
| Cold start | 51 | 66.67 | 82.68 | -0.3950 |
| Non-cold-start | 613 | 65.48 | 83.65 | 0.2552 |

Relative to the historical-average baseline, Random Forest improved MAE by
8.19 yards, RMSE by 13.03 yards, and R² by 0.2231, and won on MAE. Relative to
Linear Regression, it improved MAE by 0.43 yards but changed RMSE by -1.57
yards and R² by -0.0248: Linear Regression remains better on RMSE and R².
These comparisons use `comparison - Random Forest` for MAE/RMSE and
`Random Forest - comparison` for R², so positive values favor Random Forest.

Random Forest's overall MAE advantage over Linear Regression is narrow. It
substantially improved cold-start MAE (66.67 versus 76.30), whereas Linear
Regression was slightly better for non-cold-start MAE (65.14 versus 65.48).
Linear Regression's lower RMSE suggests better control of large misses.
Random Forest is the MAE leader; Linear Regression remains the RMSE and R²
leader. No final model has been selected. These results apply to all eligible
QB rows, including backups and low-volume passers, and do not establish
profitability or production readiness.

| Feature | Importance |
| --- | ---: |
| qb_last3_passing_attempts_avg | 0.165662 |
| qb_season_passing_yards_avg | 0.157385 |
| qb_last3_passing_yards_avg | 0.127452 |
| defense_season_passing_attempts_allowed_avg | 0.087336 |
| defense_last3_passing_yards_allowed_avg | 0.081147 |
| defense_season_passing_yards_allowed_avg | 0.077051 |
| qb_season_history_games | 0.067009 |
| qb_season_passing_attempts_avg | 0.065038 |
| defense_season_history_games | 0.049062 |
| defense_matchup_rank | 0.048607 |

These are impurity-based Random Forest importances. They are not causal
effects and do not measure a feature's standalone value. Recent passing
attempts were highest-ranked in this fitted model. Defense-feature contribution
must be tested later through controlled ablation.

### Gradient Boosting Validation

The fixed Gradient Boosting benchmark uses
`sklearn.ensemble.GradientBoostingRegressor` with one predetermined,
untuned configuration:

* `loss="squared_error"`
* `learning_rate=0.05`
* `n_estimators=200`
* `subsample=1.0`
* `criterion="friedman_mse"`
* `min_samples_split=2`
* `min_samples_leaf=1`
* `max_depth=3`
* `max_features=None`
* `random_state=42`

No tuning or parameter sweep occurred, and these parameters were not selected
using 2025 results. The model trained on 2021-2024 only, was evaluated on 2025
only, and remained fixed throughout 2025 while point-in-time features updated
by game week. It uses the exact same 14 public predictive features as the
completed models: training-only median imputation for permitted missing
numeric history, validated/pass-through missing-history flags, no scaling, and
no row dropping. Identifiers, metadata, target values, home/away, and
sportsbook lines are not inputs. The fitted pipeline keeps preprocessing and
the estimator together.

| Group | Rows | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall | 664 | 64.27 | 81.66 | 0.3662 |
| Cold start | 51 | 66.04 | 79.02 | -0.2743 |
| Non-cold-start | 613 | 64.12 | 81.88 | 0.2863 |

Positive improvement favors Gradient Boosting: comparison MAE minus Gradient
Boosting MAE, comparison RMSE minus Gradient Boosting RMSE, and Gradient
Boosting R² minus comparison R².

| Comparison | MAE improvement | RMSE improvement | R² improvement |
| --- | ---: | ---: | ---: |
| Historical average | +9.4944 yards | +14.9417 yards | +0.2532 |
| Linear Regression | +1.7308 yards | +0.3389 yards | +0.0053 |
| Random Forest | +1.2999 yards | +1.9123 yards | +0.0300 |

Gradient Boosting is the current validation leader on MAE, RMSE, and R² for
the established eligible-QB population. Its MAE is about 64 yards per
QB-game, and it improves cold-start and non-cold-start MAE relative to the
earlier models. Its advantage over Linear Regression is real but modest, it
has not been tuned, and no final model has been selected. The population still
includes backups and low-volume passers; these results do not establish betting
profitability or production readiness.

The installed scikit-learn version emitted a non-blocking deprecation warning
for the explicitly specified `criterion` parameter. No numerical or modeling
warning occurred.

| Feature | Importance |
| --- | ---: |
| qb_last3_passing_yards_avg | 0.211636 |
| qb_last3_passing_attempts_avg | 0.202901 |
| qb_season_passing_yards_avg | 0.190369 |
| qb_missing_history | 0.136354 |
| qb_season_history_games | 0.074918 |
| defense_season_passing_attempts_allowed_avg | 0.032941 |
| qb_season_passing_attempts_avg | 0.031918 |
| defense_season_passing_yards_allowed_avg | 0.030403 |
| qb_last3_history_games | 0.029363 |
| defense_last3_passing_yards_allowed_avg | 0.024767 |

These are impurity-based feature importances, not causal effects or evidence
of a feature's standalone value. Controlled ablation is still required. Recent
QB form and workload dominate this fitted model's ranking; defensive features
remain present but collectively rank below the leading QB features.

### XGBoost Validation

The fixed XGBoost benchmark uses XGBoost 3.2.0 and `XGBRegressor` with one
predetermined, untuned CPU configuration:

* `objective="reg:squarederror"`
* `n_estimators=300`
* `learning_rate=0.05`
* `max_depth=3`
* `min_child_weight=1`
* `gamma=0.0`
* `subsample=0.8`
* `colsample_bytree=0.8`
* `reg_alpha=0.0`
* `reg_lambda=1.0`
* `tree_method="hist"`
* `random_state=42`
* `n_jobs=1`
* `verbosity=0`

It uses exactly the same 14 public predictive features as the completed
models. Numeric features use training-only median imputation; existing
missing-history flags are validated and passed through unchanged. No scaling
or row dropping occurs. Identifiers, names, team/opponent labels, home/away,
targets, sportsbook lines, and outcome-derived information are excluded. Cold
start remains exactly a missing `qb_season_passing_yards_avg`.

The model trained on 2021-2024 only and was evaluated only on the fixed 2025
validation population. It was not fitted using validation outcomes. The 2026
test partition remained outcome-blind and was not transformed, predicted, or
scored.

| Group | Rows | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall | 664 | 64.6208 | 82.0804 | 0.3596 |
| Cold start | 51 | 67.8579 | 81.0890 | -0.3420 |
| Non-cold-start | 613 | 64.3515 | 82.1624 | 0.2813 |

Positive improvements favor XGBoost: comparison MAE minus XGBoost MAE,
comparison RMSE minus XGBoost RMSE, and XGBoost R² minus comparison R².

| Compared with | MAE improvement | RMSE improvement | R² improvement |
| --- | ---: | ---: | ---: |
| Historical average | +9.1414 | +14.5214 | +0.2466 |
| Linear Regression | +1.3777 | -0.0814 | -0.0013 |
| Random Forest | +0.9469 | +1.4920 | +0.0235 |
| Gradient Boosting | -0.3531 | -0.4203 | -0.0065 |

XGBoost improves substantially over the historical baseline and beats Linear
Regression and Random Forest on MAE. It is slightly worse than Linear
Regression on RMSE and R², and slightly worse than sklearn Gradient Boosting
on all three aggregate metrics. sklearn Gradient Boosting remains the current
validation leader. This is not final model selection; the 2026 holdout must
remain untouched.

XGBoost importance uses gain. XGBoost 3.2 exposes raw gain values that are not
normalized, so the report normalizes finite, nonnegative gains to sum to one
for reporting only. That normalization does not affect training or
predictions. Gain importance is not a causal effect or standalone evidence
that a feature is valuable.

| Feature | Normalized gain importance |
| --- | ---: |
| qb_last3_history_games | 0.208170 |
| qb_season_history_games | 0.163006 |
| qb_last3_passing_yards_avg | 0.137035 |
| qb_last3_passing_attempts_avg | 0.134866 |
| qb_season_passing_yards_avg | 0.116069 |
| qb_season_passing_attempts_avg | 0.033498 |
| defense_season_passing_yards_allowed_avg | 0.032460 |
| defense_season_history_games | 0.031983 |
| defense_last3_history_games | 0.030061 |
| defense_season_passing_attempts_allowed_avg | 0.029140 |

### Validation Comparison and Error Analysis

`football/training/qb_passing_yards_model_comparison.py` provides
`compare_qb_passing_yards_models(dataset_split)`. It fits all five completed
models using only `split.train`, then predicts and analyzes only canonicalized
`split.validation`. It never accesses `split.test`. Every model is evaluated
on the same validation keys: `season`, `week`, `game_id`, and `player_id`.
The result is an immutable aggregate report; it does not retain input
DataFrames or row-level records.

The comparison uses signed error defined as `prediction - actual`, so a
positive mean signed error denotes average overprediction. Cold start remains
exactly a missing `qb_season_passing_yards_avg`. The component reports overall
and cold-start diagnostics, point-in-time feature slices, error distributions,
row-level closest-prediction counts, pairwise MAE differences, and prediction
disagreement. It neither tunes models nor selects one.

The comparison used source-history seasons 2020-2026 and modeling targets
2021-2026: 2,615 training rows from 2021-2024 and 664 validation rows from
2025. The test boundary is 2026. No 2026 rows were accessed, transformed,
predicted, scored, or outcome-inspected by either the comparison component or
its temporary validation runner.

| Model | MAE | RMSE | R² | Mean signed error |
| --- | ---: | ---: | ---: | ---: |
| Historical average | 73.76 | 96.60 | 0.1130 | +5.82 |
| Linear Regression | 66.00 | 82.00 | 0.3609 | +5.48 |
| Random Forest | 65.57 | 83.57 | 0.3361 | +3.86 |
| sklearn Gradient Boosting | 64.27 | 81.66 | 0.3662 | +3.92 |
| XGBoost | 64.62 | 82.08 | 0.3596 | +5.84 |

All five models overpredicted slightly on average. sklearn Gradient Boosting
remains the aggregate validation leader on MAE, RMSE, and R². This is not
final model selection.

| Model | Cold MAE / RMSE / R² / bias | Non-cold MAE / RMSE / R² / bias |
| --- | --- | --- |
| Historical average | 150.72 / 161.10 / -4.2968 / +145.10 | 67.36 / 89.16 / 0.1538 / -5.77 |
| Linear Regression | 76.30 / 82.67 / -0.3948 / +39.80 | 65.14 / 81.94 / 0.2852 / +2.62 |
| Random Forest | 66.67 / 82.68 / -0.3950 / +22.84 | 65.48 / 83.65 / 0.2552 / +2.28 |
| sklearn Gradient Boosting | 66.04 / 79.02 / -0.2743 / +26.87 | 64.12 / 81.88 / 0.2863 / +2.02 |
| XGBoost | 67.86 / 81.09 / -0.3420 / +27.16 | 64.35 / 82.16 / 0.2813 / +4.07 |

There are 51 cold-start validation rows and 613 non-cold-start rows.
Gradient Boosting has the best cold-start and non-cold-start aggregate metrics
among the completed models. Cold-start R² is negative for every model, making
it a difficult subgroup; that does not establish that its predictions contain
no value. These subgroup results are descriptive, not causal.

AE means absolute error in passing yards. Under and over counts use a narrow
`1e-12` approximately-exact tolerance.

| Model | Median AE | 75th AE | 90th AE | Max AE | Under | Over | Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical average | 57.00 | 107.95 | 171.98 | 320.00 | 328 | 327 | 9 |
| Linear Regression | 57.76 | 87.96 | 133.62 | 281.08 | 311 | 353 | 0 |
| Random Forest | 53.42 | 91.01 | 138.51 | 304.45 | 312 | 352 | 0 |
| sklearn Gradient Boosting | 53.75 | 89.20 | 135.31 | 295.34 | 306 | 358 | 0 |
| XGBoost | 51.63 | 89.59 | 133.33 | 286.78 | 303 | 361 | 0 |

XGBoost has the lowest median and 90th-percentile absolute error, while Linear
Regression has the lowest maximum absolute error. Gradient Boosting has the
best aggregate MAE/RMSE despite not leading every distribution statistic.
These differences support additional population and feature analysis rather
than immediate final selection.

For a row-level win, a model must have the smallest absolute error on that
validation row. Ties within `1e-12` receive a win for every tied model; wins
therefore do not replace MAE, RMSE, R², or distribution diagnostics.

| Model | Wins |
| --- | ---: |
| Historical average | 182 |
| Linear Regression | 120 |
| Random Forest | 169 |
| sklearn Gradient Boosting | 92 |
| XGBoost | 101 |

The historical baseline can be closest on many individual rows while remaining
worst overall because its losing errors are substantially larger. Its
cold-start MAE, 90th-percentile error, and maximum error show that instability.

Pairwise MAE difference is defined as the first model's MAE minus the second
model's MAE, so a positive value favors the second model. Mean absolute
prediction disagreement ranges from 8.09 yards for Gradient Boosting versus
XGBoost to 34.62 yards for the historical baseline versus Random Forest. Under
that sign convention, Gradient Boosting improves MAE over Linear Regression by
1.73 yards and over XGBoost by 0.35 yards.

The report defines validation slices only from point-in-time pregame features:

* Cold versus non-cold starts.
* QB season-history depth: 0, 1-3, 4-8, and 9+ games.
* QB recent-history depth: 0, 1, 2, and 3 games.
* Defense season-history depth: 0, 1-3, 4-8, and 9+ games.
* Defensive matchup-rank tiers: 1-10, 11-22, 23-32, and missing.

XGBoost has the lowest MAE for QBs with one to three season-history games,
when one recent-history game is available, and in the defense four-to-eight-
game history slice. Gradient Boosting remains best in aggregate. These are
descriptive validation findings, not causal conclusions, evidence of future
performance, or a specialized routing strategy.

### Likely-Primary-QB Population Analysis

`football/training/qb_likely_primary_population_analysis.py` provides
`audit_qb_primary_qb_feasibility()` and
`analyze_qb_likely_primary_population(dataset_split, schedule_rows,
depth_chart_snapshots)`. The feasibility audit found that dated nflverse
depth-chart snapshots support a genuine pregame candidate selector for the
covered 2025-2026 data. Legacy 2020-2024 weekly depth charts and weekly
rosters lack a reliable snapshot timestamp, so they are excluded from
selection. Weekly player statistics and completed-game primary-passer logic
are also rejected because they reveal target-game participation or outcomes.
This does not imply that every historical season supports the dated-depth-chart
selector.

For each scheduled team-game, the selector matches QB candidates from
same-season depth-chart snapshots dated strictly before the scheduled game
date; same-day snapshots are excluded. It selects the lowest numeric depth
rank and breaks equal-rank ties deterministically by `player_id`, selecting at
most one likely-primary candidate per team-game. Where no usable pregame
evidence exists, the classification is `Unknown`. Target-game passing attempts,
passing yards, starts, snaps, plays, participation, and outcomes are never
used for selection.

The candidate selector is deployable because its inputs are pregame dated
snapshots. The scored likely-primary population is nevertheless a retrospective
subset of observed QB passing rows: a selected candidate who did not appear in
the target passing dataset has no passing-yards outcome in the existing scored
dataset. Consequently, this analysis does not evaluate every selected candidate
and must not be described as complete deployment performance. It also does not
show that excluding difficult rows improves a predictive model.

Models were trained only on the 2,615 2021-2024 training rows, and population
scoring used only the 664-row 2025 validation partition. No 2026 test rows,
predictions, outcomes, metrics, or structural counts were accessed or reported.
Although dated depth-chart source support extends into 2026, no 2026 analysis
result is presented here.

| Coverage item | Value |
| --- | ---: |
| Scheduled team-games | 544 |
| Team-games with zero selected candidates | 0 |
| Team-games with exactly one selected candidate | 544 |
| Team-games with multiple selected candidates | 0 |
| Candidate coverage | 100% |
| Observed likely-primary QB rows | 495 |
| Observed not-likely-primary QB rows | 169 |
| Unknown observed rows | 0 |

The 544 scheduled team-games count each team side of a game separately. The
495 observed likely-primary rows do not mean that only 495 candidates were
selected: some selected candidates did not appear in the observed QB
passing-yards dataset. The 169 observed not-likely-primary rows are other
observed QB passing rows, not 169 team-games without a primary selection.

Each cell in the following table is **MAE / RMSE / R² / mean signed error**;
signed error remains `prediction - actual`.

| Model | Full population | Likely primary | Not likely primary |
| --- | --- | --- | --- |
| Historical average | 73.76 / 96.60 / 0.1130 / +5.82 | 64.36 / 82.42 / -0.1891 / -5.70 | 101.31 / 129.49 / -0.6863 / +39.53 |
| Linear Regression | 66.00 / 82.00 / 0.3609 / +5.48 | 59.88 / 75.50 / 0.0023 / -4.61 | 83.92 / 98.61 / 0.0222 / +35.01 |
| Random Forest | 65.57 / 83.57 / 0.3361 / +3.86 | 60.37 / 76.03 / -0.0119 / -4.18 | 80.79 / 102.52 / -0.0569 / +27.39 |
| sklearn Gradient Boosting | 64.27 / 81.66 / 0.3662 / +3.92 | 59.11 / 74.86 / 0.0190 / -4.31 | 79.39 / 98.92 / 0.0160 / +28.05 |
| XGBoost | 64.62 / 82.08 / 0.3596 / +5.84 | 59.53 / 75.16 / 0.0113 / -2.03 | 79.54 / 99.63 / 0.0018 / +28.88 |

sklearn Gradient Boosting has the lowest MAE in the full, observed
likely-primary, and observed not-likely-primary populations: 59.11 yards for
the likely-primary rows and 79.39 yards for the not-likely-primary rows. Its
not-likely-primary signed bias is +28.05 yards, indicating average
overprediction in that validation subgroup; its likely-primary signed bias is
-4.31 yards, indicating slight average underprediction. All five models have
substantially larger errors and positive signed bias on observed
not-likely-primary rows. This is evidence that role and expected opportunity
deserve additional study, not evidence that population filtering improves the
underlying model, a causal conclusion, or final model selection. Near-zero or
negative subgroup R² values likewise do not establish that predictions have no
value.

Row-level wins use the established `1e-12` absolute-error tie tolerance; every
tied model receives a win. A win means only that a model was closest on that
row and does not replace MAE, RMSE, R², bias, or distribution diagnostics.

| Model | Likely-primary wins | Not-likely-primary wins |
| --- | ---: | ---: |
| Historical average | 122 | 60 |
| Linear Regression | 98 | 22 |
| Random Forest | 124 | 45 |
| sklearn Gradient Boosting | 75 | 17 |
| XGBoost | 76 | 25 |

The historical baseline can accumulate many wins while performing worse overall
because its losing errors are larger.

### QB-History and Defense Feature Ablation

`football/training/qb_passing_yards_feature_ablation.py` provides a controlled,
leakage-safe test of whether the existing point-in-time opponent-defense
features add value beyond QB history. It holds training rows, validation rows,
estimator parameters, preprocessing behavior, random seeds, and evaluation
methods constant; only the selected feature group changes.

The three fixed feature groups are:

* **QB_HISTORY_ONLY** (7): `qb_season_passing_yards_avg`,
  `qb_last3_passing_yards_avg`, `qb_season_passing_attempts_avg`,
  `qb_last3_passing_attempts_avg`, `qb_season_history_games`,
  `qb_last3_history_games`, and `qb_missing_history`.
* **DEFENSE_ONLY** (7): `defense_season_passing_yards_allowed_avg`,
  `defense_last3_passing_yards_allowed_avg`,
  `defense_season_passing_attempts_allowed_avg`,
  `defense_season_history_games`, `defense_last3_history_games`,
  `defense_missing_history`, and `defense_matchup_rank`.
* **QB_AND_DEFENSE**: the dataset builder's complete canonical 14-feature
  public contract.

No identifiers, names, team/opponent labels, home/away, targets, sportsbook
lines, depth-chart fields, target-game statistics, or outcome-derived values
are model inputs. Every variant uses training-only median imputation; only the
missing-history flags belonging to its selected group are included and those
booleans pass through unchanged. No rows are dropped, and tree models are not
scaled. The all-14-feature variants reproduce the completed public model
predictions and metrics within a `1e-10` tolerance.

The ablation used source-history seasons 2020-2026 and modeling targets
2021-2026: 2,615 training rows from 2021-2024 and 664 validation rows from
2025. The test boundary is 2026. No test-partition property or 2026 row was
read, transformed, predicted, scored, summarized, counted, or inspected by the
component.

Signed error is `prediction - actual`, so positive values indicate average
overprediction.

| Model | Feature group | MAE | RMSE | R² | Mean signed error |
| --- | --- | ---: | ---: | ---: | ---: |
| Linear Regression | QB history | 66.6256 | 82.6567 | 0.3506 | +4.7759 |
| Linear Regression | Defense only | 81.5523 | 102.8953 | -0.0064 | +10.0747 |
| Linear Regression | QB + defense | 65.9986 | 81.9990 | 0.3609 | +5.4785 |
| Random Forest | QB history | 68.4121 | 86.3374 | 0.2915 | +4.8715 |
| Random Forest | Defense only | 85.1389 | 107.5643 | -0.0998 | +19.1236 |
| Random Forest | QB + defense | 65.5677 | 83.5724 | 0.3361 | +3.8584 |
| sklearn Gradient Boosting | QB history | 65.8575 | 82.9298 | 0.3463 | +5.8013 |
| sklearn Gradient Boosting | Defense only | 82.6331 | 104.0549 | -0.0292 | +11.1341 |
| sklearn Gradient Boosting | QB + defense | 64.2678 | 81.6601 | 0.3662 | +3.9248 |
| XGBoost | QB history | 65.6915 | 83.0583 | 0.3443 | +5.0999 |
| XGBoost | Defense only | 82.1412 | 104.2369 | -0.0328 | +12.9219 |
| XGBoost | QB + defense | 64.6208 | 82.0804 | 0.3596 | +5.8399 |

For the adding-defense comparison, MAE improvement is QB-history-only MAE
minus QB-and-defense MAE; RMSE improvement uses the same direction; R²
improvement is QB-and-defense R² minus QB-history-only R². Positive values
therefore favor adding defense.

| Model | MAE improvement | RMSE improvement | R² improvement | 95% game-cluster MAE interval |
| --- | ---: | ---: | ---: | --- |
| Linear Regression | +0.6270 | +0.6577 | +0.0103 | -0.1196 to +1.3484 |
| Random Forest | +2.8444 | +2.7650 | +0.0447 | +0.9402 to +4.7569 |
| sklearn Gradient Boosting | +1.5897 | +1.2697 | +0.0199 | +0.4833 to +2.6793 |
| XGBoost | +1.0707 | +0.9778 | +0.0153 | -0.2701 to +2.3815 |

Predictions are paired on the same validation rows. The uncertainty diagnostic
resamples `game_id` clusters, so all QB rows in a game stay together: 2,000
replicates, random seed 42, a 95% percentile interval, and NumPy's `linear`
percentile method. It estimates uncertainty in the paired validation MAE
improvement; it does not establish causality, formal universal statistical
significance, or guaranteed future performance.

Defense-only models are materially worse than QB-history-only or combined
models for every estimator, so QB history contains most of the predictive
information among the current features. Adding defense improves point-estimate
MAE, RMSE, and R² for all four learned estimators. The Random Forest and
Gradient Boosting intervals are entirely above zero, providing validation
evidence that defense features reduce MAE for those estimators. The Linear
Regression and XGBoost intervals include zero, making their added-defense MAE
benefit uncertain on this validation sample; this does not mean the defense
features are useless. sklearn Gradient Boosting with all 14 features remains
the aggregate validation leader. This is not final model selection.

Cold start remains exactly a missing `qb_season_passing_yards_avg`.

| Model | Cold-start MAE | Non-cold-start MAE |
| --- | ---: | ---: |
| Linear Regression | 76.3004 | 65.1415 |
| Random Forest | 66.6728 | 65.4758 |
| sklearn Gradient Boosting | 66.0370 | 64.1206 |
| XGBoost | 67.8579 | 64.3515 |

### 2026 Test-Set Policy

> **2026 is held out for final model evaluation.** The original smoke-run
> snapshot had 73 test rows across 42 QBs and 31 games in Weeks 1-2. During the
> later Linear Regression validation run, the live public feed had updated to
> 76 rows across 43 QBs and 32 games in the same weeks. These are
> time-dependent data-availability snapshots, not dataset implementation
> changes. During the Gradient Boosting run, it had reached 112 structural rows
> through Week 3, another availability snapshot rather than a modeling change.

No 2026 rows were transformed or predicted during Linear Regression, Random
Forest, sklearn Gradient Boosting, XGBoost, or validation-comparison work. No
2026 outcomes, predictions, residuals, or metrics were inspected. Feed growth
did not affect the fixed 2021-2024 training or 2025 validation results. The
2026 holdout remains locked until final model selection and must not be used
while selecting preprocessing, features, populations, hyperparameters, or
models. Test structural and feature-availability diagnostics remain allowed.

### Results Ledger

| Model | Validation population | MAE | RMSE | R² | Status |
| --- | --- | ---: | ---: | ---: | --- |
| Historical average | All 2025 eligible QB rows | 73.76 | 96.60 | 0.1130 | Official baseline |
| Linear Regression | All 2025 eligible QB rows | 66.00 | 82.00 | 0.3609 | Previous benchmark |
| Random Forest | All 2025 eligible QB rows | 65.57 | 83.57 | 0.3361 | Previous MAE leader |
| XGBoost | All 2025 eligible QB rows | 64.6208 | 82.0804 | 0.3596 | Initial benchmark complete |
| sklearn Gradient Boosting | All 2025 eligible QB rows | 64.2678 | 81.6601 | 0.3662 | Current validation leader |

### Completed QB Modeling Stages

1. [x] Training-only preprocessing for missing history.
2. [x] Initial Linear Regression using existing leakage-safe features.
3. [x] Fixed Random Forest benchmark.
4. [x] Fixed Gradient Boosting benchmark.
5. [x] Fixed XGBoost benchmark.
6. [x] Fixed initial general-purpose model benchmarks.
7. [x] Consistent validation comparison and error analysis.
8. [x] Dated-depth-chart feasibility audit and likely-primary-QB population
   analysis.
9. [x] Controlled QB-history/defense feature ablation with a paired
   game-cluster bootstrap diagnostic.

### Next QB Modeling Stages

1. [ ] Defensive-strength representation analysis: determine whether continuous
   defense metrics, `defense_matchup_rank`, fixed rank tiers, or combinations
   provide the most useful validation signal.
2. [ ] Opponent-adjusted QB form.
3. [ ] Defensive/offensive style-data feasibility audit, including blitz,
   pressure, coverage, and offensive-style availability.
4. [ ] Blitz, pressure, coverage, personnel, motion, and offensive-style
   analysis where reliable historical data exists.
5. [ ] QB-versus-defense and defense-versus-offense style experiments.
6. [ ] Defensive/offensive clustering and interaction features.
7. [ ] Final validation-based model selection.
8. [ ] One-time 2026 holdout evaluation.
9. [ ] Model persistence and weekly inference.
10. [ ] Streamlit integration.
11. [ ] Sportsbook comparison and historical backtesting.
12. [ ] RB rushing-yards modeling.

Defensive tiers are not fixed in advance. Continuous values, thirds,
quartiles, top/bottom groups, or data-derived clusters may be compared using
chronological validation.

Paid historical sportsbook lines are not currently part of model results. They
may be evaluated later for projection-versus-line backtesting.

Offensive-style work may include measurable versions of West Coast or
Shanahan-style tendencies, but no subjective or unsupported scheme labels have
been implemented.

No current result makes claims from blitz, pressure, man/zone, personnel,
motion, or offensive-scheme data; their availability and historical reliability
must be audited before they can support a future experiment.

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
