"""Point-in-time opponent-adjusted QB passing-yard history.

An adjusted observation is a completed QB's passing yards minus the opposing
defense's *pregame* season passing-yards-allowed average.  The allowance is a
team-game primary-QB measure, while the result is an individual-QB measure, so
this is a descriptive ``over_opponent_pregame_allowance`` feature—not an
expected-yards model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from football.data.build_quarterback_dataset import build_quarterback_dataset
from football.features.defense_vs_quarterbacks import build_defense_vs_primary_quarterback_logs
from football.features.primary_quarterbacks import identify_primary_quarterbacks
from football.training.build_qb_passing_yards_dataset import (
    FEATURE_COLUMNS, TARGET_COLUMN,
    TARGET_KEY,
    _attach_schedule_context,
    _history_cutoff,
    _normalize_schedule_context,
    _prior_rows,
)
from football.training.qb_passing_yards_feature_ablation import (
    BOOTSTRAP_SEED, DEFAULT_BOOTSTRAP_REPLICATES, ESTIMATOR_NAMES,
    FeatureAblationPopulationMetrics, _canonical_frame, _fit_predict,
    _paired_bootstrap, _population_metrics, _estimator, _finite,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit

ADJUSTED_FORM_FEATURE_COLUMNS = (
    "qb_season_passing_yards_over_opponent_allowance_avg",
    "qb_last3_passing_yards_over_opponent_allowance_avg",
    "qb_season_opponent_adjusted_history_games",
    "qb_last3_opponent_adjusted_history_games",
    "qb_missing_opponent_adjusted_history",
)
OFFICIAL_FEATURES = "OFFICIAL_FEATURES"
OFFICIAL_PLUS_OPPONENT_ADJUSTED_FORM = "OFFICIAL_PLUS_OPPONENT_ADJUSTED_FORM"


@dataclass(frozen=True)
class OpponentAdjustedFormVariant:
    estimator_name: str
    variant_name: str
    feature_columns: tuple[str, ...]
    training_row_count: int
    validation_row_count: int
    validation: FeatureAblationPopulationMetrics
    cold_start: FeatureAblationPopulationMetrics
    non_cold_start: FeatureAblationPopulationMetrics
    adjusted_history_available: FeatureAblationPopulationMetrics
    adjusted_history_missing: FeatureAblationPopulationMetrics


@dataclass(frozen=True)
class OpponentAdjustedFormComparison:
    estimator_name: str
    mae_improvement_from_opponent_adjusted_form: float
    rmse_improvement_from_opponent_adjusted_form: float
    r_squared_improvement_from_opponent_adjusted_form: float
    bootstrap_point_estimate: float
    bootstrap_lower_bound: float
    bootstrap_upper_bound: float
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_cluster_count: int
    percentile_method: str


@dataclass(frozen=True)
class QBPassingYardsOpponentAdjustedFormAnalysis:
    validation_row_count: int
    estimator_variants: tuple[OpponentAdjustedFormVariant, ...]
    comparisons: tuple[OpponentAdjustedFormComparison, ...]


def analyze_qb_passing_yards_opponent_adjusted_form(
    dataset_split: QBPassingYardsDatasetSplit, *, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
) -> QBPassingYardsOpponentAdjustedFormAnalysis:
    """Compare official inputs with adjusted form using train/validation only."""
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be a positive integer")
    required = [*FEATURE_COLUMNS, *ADJUSTED_FORM_FEATURE_COLUMNS]
    train = _canonical_adjusted(dataset_split.train, "training", required)
    validation = _canonical_adjusted(dataset_split.validation, "validation", required)
    actual = validation[TARGET_COLUMN].to_numpy(dtype=float)
    cold = validation["qb_season_passing_yards_avg"].isna().to_numpy()
    adjusted_missing = validation["qb_missing_opponent_adjusted_history"].to_numpy(dtype=bool)
    variants, comparisons = [], []
    for estimator in ESTIMATOR_NAMES:
        official = _fit_predict(estimator, tuple(FEATURE_COLUMNS), train, validation)
        enhanced_columns = (*FEATURE_COLUMNS, *ADJUSTED_FORM_FEATURE_COLUMNS)
        enhanced = _fit_predict_adjusted(estimator, enhanced_columns, train, validation)
        a = _adjusted_variant(estimator, OFFICIAL_FEATURES, tuple(FEATURE_COLUMNS), train, validation, actual, official, cold, adjusted_missing)
        b = _adjusted_variant(estimator, OFFICIAL_PLUS_OPPONENT_ADJUSTED_FORM, enhanced_columns, train, validation, actual, enhanced, cold, adjusted_missing)
        variants.extend((a, b))
        boot = _paired_bootstrap(estimator, validation["game_id"].to_numpy(), actual, official, enhanced, bootstrap_replicates)
        comparisons.append(OpponentAdjustedFormComparison(estimator, a.validation.metrics.mae-b.validation.metrics.mae, a.validation.metrics.rmse-b.validation.metrics.rmse, b.validation.metrics.r_squared-a.validation.metrics.r_squared, boot.point_estimate, boot.lower_bound, boot.upper_bound, boot.replicate_count, boot.seed, boot.cluster_count, boot.percentile_method))
    return QBPassingYardsOpponentAdjustedFormAnalysis(len(validation), tuple(variants), tuple(comparisons))


def _canonical_adjusted(frame: pd.DataFrame, name: str, required: list[str]) -> pd.DataFrame:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{name.capitalize()} frame is missing adjusted-form columns: {missing}")
    result = _canonical_frame(frame, name, require_target=True)
    for column in ADJUSTED_FORM_FEATURE_COLUMNS[:4]:
        numeric = pd.to_numeric(result[column], errors="coerce")
        if (~result[column].isna() & (numeric.isna() | ~np.isfinite(numeric))).any():
            raise ValueError(f"{name.capitalize()} frame has invalid adjusted-form values")
    for column in ADJUSTED_FORM_FEATURE_COLUMNS[2:4]:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or values.lt(0).any() or values.mod(1).ne(0).any():
            raise ValueError(f"{name.capitalize()} frame has invalid adjusted-history counts")
    flag = result[ADJUSTED_FORM_FEATURE_COLUMNS[4]]
    if not pd.api.types.is_bool_dtype(flag) or flag.isna().any() or not flag.eq(result[ADJUSTED_FORM_FEATURE_COLUMNS[2]].eq(0)).all():
        raise ValueError(f"{name.capitalize()} frame has inconsistent adjusted-history flag")
    return result


def _fit_predict_adjusted(estimator, columns, train, validation):
    binary = tuple(c for c in columns if c.endswith("_missing_history") or c == ADJUSTED_FORM_FEATURE_COLUMNS[4])
    numeric = tuple(c for c in columns if c not in binary)
    if any(train[c].isna().all() for c in numeric):
        raise ValueError("Training frame has selected numeric features with no usable values")
    pipe = Pipeline([("preprocessing", ColumnTransformer([("numeric", SimpleImputer(strategy="median"), list(numeric)), ("binary", "passthrough", list(binary))], verbose_feature_names_out=False)), ("estimator", _estimator(estimator))])
    pipe.fit(train.loc[:, columns].copy(deep=True), _finite(train[TARGET_COLUMN], "training target_passing_yards"))
    values = np.asarray(pipe.predict(validation.loc[:, columns].copy(deep=True)), dtype=float).reshape(-1)
    if len(values) != len(validation) or not np.isfinite(values).all():
        raise ValueError("Adjusted-form predictions must be finite and aligned")
    return values


def _adjusted_variant(estimator, name, columns, train, validation, actual, prediction, cold, adjusted_missing):
    all_rows = np.ones(len(validation), dtype=bool)
    return OpponentAdjustedFormVariant(estimator, name, columns, len(train), len(validation), _population_metrics(validation, actual, prediction, all_rows), _population_metrics(validation, actual, prediction, cold), _population_metrics(validation, actual, prediction, ~cold), _population_metrics(validation, actual, prediction, ~adjusted_missing), _population_metrics(validation, actual, prediction, adjusted_missing))


def build_qb_opponent_adjusted_form_features(
    quarterback_games: pd.DataFrame, schedule_rows: pd.DataFrame
) -> pd.DataFrame:
    """Build five leakage-safe adjusted-form features keyed by a QB target game.

    Every week is evaluated before outcomes from that week are eligible.  Week
    one follows the existing QB-form convention and uses the preceding season;
    later weeks use only the same season's strictly earlier weeks.
    """
    qbs = _attach_schedule_context(
        build_quarterback_dataset(quarterback_games),
        _normalize_schedule_context(schedule_rows),
    )
    qbs = qbs.copy(deep=True)
    qbs["passing_yards"] = pd.to_numeric(qbs["passing_yards"], errors="coerce")
    qbs.loc[~np.isfinite(qbs["passing_yards"]), "passing_yards"] = np.nan
    if qbs.duplicated(TARGET_KEY).any():
        raise ValueError("Conflicting duplicate QB-game keys are not permitted")
    defense_logs = build_defense_vs_primary_quarterback_logs(
        identify_primary_quarterbacks(qbs)
    )
    if defense_logs.duplicated(["season", "week", "game_id", "defense"]).any():
        raise ValueError("Conflicting duplicate defense team-game keys are not permitted")
    observations = []
    for row in qbs.sort_values(TARGET_KEY, kind="mergesort").itertuples(index=False):
        season, week = _history_cutoff(row.season, row.week)
        prior_defense = _prior_rows(defense_logs, "defense", row.opponent, season, week)
        allowance = pd.to_numeric(prior_defense["passing_yards_allowed"], errors="coerce")
        allowance = allowance[np.isfinite(allowance)].mean()
        adjusted = row.passing_yards - allowance if np.isfinite(row.passing_yards) and np.isfinite(allowance) else np.nan
        observations.append((*[getattr(row, key) for key in TARGET_KEY], adjusted))
    observed = pd.DataFrame(observations, columns=[*TARGET_KEY, "_adjusted"])
    rows = []
    for row in qbs.sort_values(TARGET_KEY, kind="mergesort").itertuples(index=False):
        season, week = _history_cutoff(row.season, row.week)
        prior = observed.loc[(observed.player_id == row.player_id) & (observed.season == season) & (observed.week < week)].sort_values(["season", "week", "game_id"], kind="mergesort")
        usable = prior.loc[np.isfinite(prior["_adjusted"])]
        last3 = usable.tail(3)
        rows.append((*[getattr(row, key) for key in TARGET_KEY], usable["_adjusted"].mean(), last3["_adjusted"].mean(), len(usable), len(last3), len(usable) == 0))
    return pd.DataFrame(rows, columns=[*TARGET_KEY, *ADJUSTED_FORM_FEATURE_COLUMNS]).sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)


def attach_qb_opponent_adjusted_form_features(
    dataset: pd.DataFrame, adjusted_features: pd.DataFrame
) -> pd.DataFrame:
    """Return an owned dataset copy with uniquely keyed adjusted features joined."""
    required_dataset = set(TARGET_KEY)
    if not required_dataset.issubset(dataset.columns):
        raise ValueError("Dataset is missing target-key columns for adjusted-form attachment")
    if dataset.duplicated(TARGET_KEY).any() or adjusted_features.duplicated(TARGET_KEY).any():
        raise ValueError("Adjusted-form attachment requires unique target keys")
    missing = set([*TARGET_KEY, *ADJUSTED_FORM_FEATURE_COLUMNS]).difference(adjusted_features.columns)
    if missing:
        raise ValueError(f"Adjusted features are missing required columns: {sorted(missing)}")
    dataset_keys = set(map(tuple, dataset.loc[:, TARGET_KEY].itertuples(index=False, name=None)))
    feature_keys = set(map(tuple, adjusted_features.loc[:, TARGET_KEY].itertuples(index=False, name=None)))
    if dataset_keys != feature_keys:
        raise ValueError("Adjusted-form feature keys must exactly match dataset target keys")
    result = dataset.merge(adjusted_features[[*TARGET_KEY, *ADJUSTED_FORM_FEATURE_COLUMNS]], on=TARGET_KEY, how="left", validate="one_to_one")
    if len(result) != len(dataset):
        raise ValueError("Adjusted-form attachment changed target-row count")
    return result.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)
