# NFL QB Passing-Yards Modeling

## 1. Overview and Current Status

This document is the technical record for the NFL QB passing-yards dataset,
experiments, and final validation-based selection. The frozen production
candidate is `sklearn.ensemble.GradientBoostingRegressor` with the official
14-feature contract. It is a candidate specification, not a saved or deployed
model.

Selection used only the 2025 validation partition. The 2026 season remains
unscored and excluded from model fitting and selection; strictly earlier
completed 2026 weeks may supply lagged live-inference features. MAE is an
average validation error, not an individual prediction guarantee; sportsbook
profitability has not been tested.

Completed components include the point-in-time dataset builder, deterministic
splitter, historical-average baseline, dataset/baseline audit, four learned
benchmarks, cross-model comparison, likely-primary population analysis,
feature ablation, defensive-representation analysis, opponent-adjusted-form
experiment, and final-selection report.

## 2. Data Sources and Modeling Population

The original smoke run used public nflverse data through existing repository
loaders (`nflreadpy` 0.1.5). Source-history seasons were 2020–2026 and modeling
target seasons were 2021–2026; 2020 supports prior-season history for 2021
Week 1. The smoke run produced 3,974 normalized QB-game rows, 3,774 normalized
schedule rows, and 3,352 modeling rows after the target-season restriction.
Target keys were unique and no loader or normalization warnings were reported.

The training population intentionally includes backups and low-volume passers:
2,615 QB rows across 1,087 games. This is a modeling-population characteristic,
not a structural audit failure.

Training-only target summary: mean 196.15 yards, sample standard deviation
103.85 (`ddof=1`), minimum -4.0, 25th percentile 135.0, median 210.0, 75th
percentile 268.0, and maximum 525.0. Missing QB season/last-three history was
188 rows (7.19%) in train and 51 rows (7.68%) in validation; missing defense
history was zero in both partitions. Expected cold starts are diagnostics, not
corruption.

## 3. Point-in-Time Feature Construction

Each row represents an eligible regular-season QB game. Week 1 histories use
only the preceding regular season; Week 2 and later use strictly earlier weeks
of the target season. The builder processes a week as a batch before that
week's outcomes enter history, preventing same-week contamination.

The raw dataset keeps permitted early-history values missing. Later model
preprocessing uses training-only imputation. Target-game passing yards and
attempts, identifiers, player/team/opponent labels, and home/away are not model
inputs.

## 4. Chronological Train/Validation/Test Boundaries

| Partition | Target period | Rows | Unique QBs | Unique games |
| --- | --- | ---: | ---: | ---: |
| Train | 2021–2024 | 2,615 | 123 | 1,087 |
| Validation | 2025 | 664 | 81 | 272 |
| Holdout | 2026 | Outcome-blind | — | — |

The split uses `validation_start=(2025, 1)` and `test_start=(2026, 1)`. The
audit verified consistent schemas, unique keys within partitions, no
cross-partition overlap, and correct chronological ordering.

## 5. Leakage-Prevention Policy

All historical features contain only information available before the target
game. Models and preprocessors fit only on training rows; all selection and
comparison work uses only the 2025 validation rows. The 2026 partition remains
unscored and excluded from model selection and fitting. For live weekly
inference only, completed games from strictly earlier 2026 weeks may supply
lagged point-in-time features; target-week outcomes are never used, and no
2026 metric, residual, or outcome evaluation is produced.

## 6. Official 14-Feature Contract

The public canonical order from `FEATURE_COLUMNS` is:

1. `qb_season_passing_yards_avg`
2. `qb_last3_passing_yards_avg`
3. `qb_season_passing_attempts_avg`
4. `qb_last3_passing_attempts_avg`
5. `qb_season_history_games`
6. `qb_last3_history_games`
7. `qb_missing_history`
8. `defense_season_passing_yards_allowed_avg`
9. `defense_last3_passing_yards_allowed_avg`
10. `defense_season_passing_attempts_allowed_avg`
11. `defense_season_history_games`
12. `defense_last3_history_games`
13. `defense_missing_history`
14. `defense_matchup_rank`

Numeric fields use training-only median imputation. Existing missing-history
booleans are validated and passed through unchanged. There is no scaling and no
row dropping; preprocessing and estimator remain together in the fitted
pipeline.

## 7. Historical-Average Baseline

The official baseline predicts the QB's pregame season passing-yards average.
For a cold start—missing `qb_season_passing_yards_avg`—it uses only the
training-target mean, 196.15 yards.

| Validation group | Rows | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Overall | 664 | 73.76 | 96.60 | 0.1130 |
| Cold start | 51 | 150.72 | 161.10 | -4.2968 |
| Non-cold-start | 613 | 67.36 | 89.16 | 0.1538 |

## 8. Completed Model Benchmarks

All learned models use the identical training population, official feature
contract, training-only preprocessing, and 2025 validation population.

| Model | MAE | RMSE | R² | Status |
| --- | ---: | ---: | ---: | --- |
| Historical average | 73.76 | 96.60 | 0.1130 | Official baseline |
| Linear Regression | 66.00 | 82.00 | 0.3609 | Benchmark |
| Random Forest | 65.57 | 83.57 | 0.3361 | Benchmark |
| XGBoost | 64.6208 | 82.0804 | 0.3596 | Benchmark |
| sklearn Gradient Boosting | 64.2678 | 81.6601 | 0.3662 | Selected candidate |

The selected fixed sklearn Gradient Boosting configuration is
`loss="squared_error"`, `learning_rate=0.05`, `n_estimators=200`,
`subsample=1.0`, `criterion="friedman_mse"`, `min_samples_split=2`,
`min_samples_leaf=1`, `max_depth=3`, `max_features=None`, and
`random_state=42`. These fixed benchmark settings were not tuned on validation.

The selected model's exact 2025 validation result is MAE **64.2678**, RMSE
**81.6601**, and R² **0.3662**. Its cold-start result is MAE **66.0370**, RMSE
**79.0168**, and R² **-0.2743**; its non-cold-start result is MAE **64.1206**,
RMSE **81.8762**, and R² **0.2863**.

Linear Regression used sklearn defaults. Random Forest used 500 trees and
`random_state=42`; XGBoost 3.2.0 used a fixed CPU configuration with 300 trees,
`tree_method="hist"`, `random_state=42`, and `n_jobs=1`.

## 9. Validation Comparison and Error Analysis

`compare_qb_passing_yards_models(dataset_split)` fits all five models on train
only and evaluates the same canonical validation keys `(season, week, game_id,
player_id)`. Signed error is `prediction - actual`, so positive bias means
average overprediction.

| Model | MAE | RMSE | R² | Mean signed error |
| --- | ---: | ---: | ---: | ---: |
| Historical average | 73.76 | 96.60 | 0.1130 | +5.82 |
| Linear Regression | 66.00 | 82.00 | 0.3609 | +5.48 |
| Random Forest | 65.57 | 83.57 | 0.3361 | +3.86 |
| sklearn Gradient Boosting | 64.27 | 81.66 | 0.3662 | +3.92 |
| XGBoost | 64.62 | 82.08 | 0.3596 | +5.84 |

| Model | Median AE | 75th AE | 90th AE | Max AE |
| --- | ---: | ---: | ---: | ---: |
| Historical average | 57.00 | 107.95 | 171.98 | 320.00 |
| Linear Regression | 57.76 | 87.96 | 133.62 | 281.08 |
| Random Forest | 53.42 | 91.01 | 138.51 | 304.45 |
| sklearn Gradient Boosting | 53.75 | 89.20 | 135.31 | 295.34 |
| XGBoost | 51.63 | 89.59 | 133.33 | 286.78 |

Gradient Boosting led aggregate MAE, RMSE, and R². XGBoost had the lowest
median and 90th-percentile absolute error; Linear Regression had the lowest
maximum absolute error. These descriptive diagnostics did not override the
aggregate selection evidence.

Row-level ties use a `1e-12` absolute-error tolerance and award every tied
model a win. The historical baseline won 182 rows, Random Forest 169, Linear
Regression 120, XGBoost 101, and Gradient Boosting 92. Win counts do not
replace aggregate metrics: the baseline's losing errors were larger. Mean
absolute prediction disagreement ranged from 8.09 yards (Gradient Boosting
versus XGBoost) to 34.62 yards (historical baseline versus Random Forest).

## 10. Cold-Start Analysis

| Model | Cold MAE / RMSE / R² / bias | Non-cold MAE / RMSE / R² / bias |
| --- | --- | --- |
| Historical average | 150.72 / 161.10 / -4.2968 / +145.10 | 67.36 / 89.16 / 0.1538 / -5.77 |
| Linear Regression | 76.30 / 82.67 / -0.3948 / +39.80 | 65.14 / 81.94 / 0.2852 / +2.62 |
| Random Forest | 66.67 / 82.68 / -0.3950 / +22.84 | 65.48 / 83.65 / 0.2552 / +2.28 |
| sklearn Gradient Boosting | 66.04 / 79.02 / -0.2743 / +26.87 | 64.12 / 81.88 / 0.2863 / +2.02 |
| XGBoost | 67.86 / 81.09 / -0.3420 / +27.16 | 64.35 / 82.16 / 0.2813 / +4.07 |

Cold-start R² was negative for every completed model. That identifies a
difficult subgroup; it does not establish that predictions have no value.

## 11. Likely-Primary-QB Population Analysis

Dated nflverse depth-chart snapshots support a genuine pregame candidate
selector for covered 2025–2026 data. It uses same-season QB snapshots strictly
before the scheduled game date, excludes same-day snapshots, selects the lowest
numeric depth rank, and breaks ties by `player_id`. It selects at most one
candidate per team-game and yields `Unknown` if evidence is insufficient.

Legacy 2020–2024 weekly depth charts and rosters lack reliable snapshot times,
so they are excluded. Target-game attempts, yards, starts, snaps, plays,
participation, and outcomes are never used. The selector is pregame deployable,
but scored results are a retrospective subset of observed QB passing rows;
selected candidates without a passing outcome cannot be scored.

For 2025 validation, 544 scheduled team-games had exactly one selected
candidate (100% coverage); 495 observed rows were likely-primary, 169 were
not-likely-primary, and zero observed rows were Unknown.

| Model | Likely-primary MAE | Not-likely-primary MAE |
| --- | ---: | ---: |
| Historical average | 64.36 | 101.31 |
| Linear Regression | 59.88 | 83.92 |
| Random Forest | 60.37 | 80.79 |
| sklearn Gradient Boosting | 59.11 | 79.39 |
| XGBoost | 59.53 | 79.54 |

This does not show that filtering difficult rows improves the underlying model
or complete deployment performance.

For the same groups, the complete diagnostics (MAE / RMSE / R² / signed error)
were:

| Model | Likely primary | Not likely primary |
| --- | --- | --- |
| Historical average | 64.36 / 82.42 / -0.1891 / -5.70 | 101.31 / 129.49 / -0.6863 / +39.53 |
| Linear Regression | 59.88 / 75.50 / 0.0023 / -4.61 | 83.92 / 98.61 / 0.0222 / +35.01 |
| Random Forest | 60.37 / 76.03 / -0.0119 / -4.18 | 80.79 / 102.52 / -0.0569 / +27.39 |
| sklearn Gradient Boosting | 59.11 / 74.86 / 0.0190 / -4.31 | 79.39 / 98.92 / 0.0160 / +28.05 |
| XGBoost | 59.53 / 75.16 / 0.0113 / -2.03 | 79.54 / 99.63 / 0.0018 / +28.88 |

## 12. QB-Versus-Defense Feature Ablation

The controlled ablation held rows, estimator settings, preprocessing, seeds,
and evaluation constant while varying seven QB-history features, seven defense
features, or all 14 features. It used paired `game_id` cluster bootstrap
resampling: 2,000 replicates, seed 42, and a 95% NumPy `linear` percentile
interval.

| Model | QB history MAE | Defense-only MAE | QB + defense MAE | Added-defense 95% MAE interval |
| --- | ---: | ---: | ---: | --- |
| Linear Regression | 66.6256 | 81.5523 | 65.9986 | -0.1196 to +1.3484 |
| Random Forest | 68.4121 | 85.1389 | 65.5677 | +0.9402 to +4.7569 |
| sklearn Gradient Boosting | 65.8575 | 82.6331 | 64.2678 | +0.4833 to +2.6793 |
| XGBoost | 65.6915 | 82.1412 | 64.6208 | -0.2701 to +2.3815 |

Defense-only variants were materially worse. QB history contains most current
predictive information, while continuous defense inputs improved every
estimator's point estimate. Random Forest and Gradient Boosting intervals were
entirely above zero; the Linear Regression and XGBoost intervals included zero.

## 13. Defensive-Strength Representation Experiment

Six fixed representations compared QB-only, continuous defense, rank-only,
tier-only, continuous-plus-rank, and continuous-plus-tier contracts. Rank 1
means the highest pregame historical mean QB passing yards allowed. Fixed tiers
were 1–10, 11–22, 23–32, and missing.

| Model | QB only | Continuous | Rank only | Tier only | Continuous + rank | Continuous + tier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear Regression | 66.63 | 66.20 | 65.99 | 65.98 | 66.00 | 66.01 |
| Random Forest | 68.41 | 65.63 | 66.39 | 66.84 | 65.57 | 65.72 |
| sklearn Gradient Boosting | 65.86 | 64.29 | 64.37 | 64.32 | 64.27 | 64.32 |
| XGBoost | 65.69 | 64.27 | 64.35 | 64.97 | 64.62 | 64.54 |

Continuous defensive performance reduced MAE versus QB-only for every
estimator (+0.42, +2.78, +1.57, and +1.42 yards). Every bootstrap interval for
numeric rank or tiers beyond continuous defense included zero. Neither showed
stable incremental validation value, although the official contract retains
numeric rank because it remains the narrow aggregate leader.

## 14. Opponent-Adjusted-Form Experiment

The adjusted historical observation was:

```text
QB passing yards − opponent defense's pregame season passing-yards allowance
```

The opponent allowance was calculated before the historical game and only
strictly earlier valid observations entered later target features. Weeks were
batched and histories reset by season. This is an *over-opponent-pregame-
allowance* measure, not a true individual expected-yards model: the defensive
allowance is a prior primary-QB/team-game measure while the target is an
individual QB result.

Gradient Boosting with the five adjusted-form fields had MAE 65.11 versus 64.27
for the official contract: improvement -0.84 yards and 95% game-cluster
interval -1.59 to -0.14. The completed feature was excluded.

## 15. Style-Data Feasibility Audit

The public-data audit found that sack and QB-hit rates and selected
formation/personnel tendencies may support future, explicitly named proxies.
Reliable full-history public coverage was not established for blitz, total
pressure, hurry, man/zone, coverage shells, motion/play-action, or named
offensive schemes. Those fields are deferred rather than inferred. Any future
clusters would be behavioral groups, not subjective coaching-scheme labels.

## 16. Final Model-Selection Rationale

`select_qb_passing_yards_final_model(dataset_split)` reuses completed Gradient
Boosting fit/evaluation behavior, fits only `split.train`, evaluates only
`split.validation`, and never reads `split.test`. sklearn Gradient Boosting
was selected because it led 2025 aggregate MAE/RMSE/R²
(64.2678 / 81.6601 / 0.3662) and had the lowest MAE in observed likely-primary
(59.11) and not-likely-primary (79.39) rows.

Continuous defense statistics were useful. Rank tiers and opponent-adjusted
form were excluded; depth-chart status, sportsbook lines, and unverified style
fields are not estimator inputs.

## 17. Known Limitations

- Validation includes backups and low-volume passers.
- Cold starts and observed non-primary appearances remain less reliable.
- Likely-primary candidate selection is separate from prediction.
- The frozen model can be persisted locally and generate unscored pregame
  weekly projections, but it is not deployed and has no Streamlit display yet.
- No sportsbook backtest exists.
- No causal conclusion follows from feature importances, slices, ablations, or
  bootstrap intervals.
- No 2026 test performance is known.

## 18. Deferred Work and Production Roadmap

1. Integrate validated unscored projection snapshots into Streamlit.
2. Explore defensible public style/proxy work if historical coverage becomes
   sufficient.
3. Perform the one-time 2026 holdout evaluation after candidate decisions are
   fixed.
4. Later compare projections with sportsbook lines, backtest, and extend the
   pipeline to RB rushing yards.

## Production Refit and Persistence

`train_qb_passing_yards_production_model(dataset_split)` refits the frozen
candidate on the completed 2021–2025 training and validation rows only. It does
not read `split.test`; 2026 remains unscored. The explicit save operation writes
a `.joblib` model and adjacent `.joblib.metadata.json` manifest containing the
canonical feature contract, fixed parameters, training boundary, versions,
SHA-256 checksum, and byte size. No real artifact is committed.

Artifacts are trusted local files only. Manifest checksum validation detects a
change relative to the manifest, but joblib/pickle deserialization is not safe
for untrusted files.

## Weekly Inference

`build_qb_passing_yards_weekly_feature_rows(...)` creates candidate feature
rows only for scheduled regular-season team-games that are strictly after an
explicit UTC as-of timestamp. Dated pregame depth charts select at most one QB
candidate per team-game; depth-chart status is metadata, never a model input.
The frozen artifact then projects only the canonical 14 point-in-time features.
2026 remains unscored and excluded from fitting and model selection.

## Weekly Projection Snapshots

`save_qb_passing_yards_weekly_projection_snapshot(...)` writes the immutable
output of weekly inference as deterministic, atomically replaced JSON. Store
generated snapshots under
`football/data/processed/qb_passing_yards_projection_snapshots/`; that
generated-data location is ignored by Git. The snapshot contains only safe
projection display identifiers, kickoff/as-of timestamps, predicted passing
yards, aggregate diagnostics, skip reasons, the canonical feature-contract
fingerprint, and the trusted artifact identity/training boundary. It excludes
raw features, targets, outcomes, residuals, models, pipelines, sportsbook
lines, and recommendations.

`load_qb_passing_yards_weekly_projection_snapshot(...)` validates the JSON
schema, counts, feature contract, timestamps, and optional expected artifact
SHA-256 without loading a model or performing inference. A checksum can detect
an identity mismatch against the manifest value, but it does not make an
untrusted joblib/pickle file safe to deserialize. Snapshots are unscored
projections, not demonstrated sportsbook edges. Streamlit display remains the
next step.
