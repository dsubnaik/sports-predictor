"""Pregame-depth-chart QB population analysis with strict test blindness.

The selector is deployable only where dated depth-chart snapshots exist.  It
uses same-season QB snapshots strictly before a scheduled game date, selects
the lowest depth rank (then player ID), and never uses target-game participation
or box-score fields.  Legacy weekly rosters/depth charts lack a snapshot time,
so they are intentionally not used.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from football.training.build_qb_passing_yards_dataset import TARGET_COLUMN, TARGET_KEY
from football.training.qb_passing_yards_baseline import (
    PREDICTION_COLUMN, RegressionMetrics, evaluate_qb_passing_yards_predictions,
    fit_qb_passing_yards_baseline, predict_qb_passing_yards_baseline,
)
from football.training.qb_passing_yards_gradient_boosting import fit_qb_passing_yards_gradient_boosting, predict_qb_passing_yards_gradient_boosting
from football.training.qb_passing_yards_linear_regression import fit_qb_passing_yards_linear_regression, predict_qb_passing_yards_linear_regression
from football.training.qb_passing_yards_model_comparison import (
    MODEL_NAMES, PERCENTILE_METHOD, PREDICTION_TIE_TOLERANCE,
)
from football.training.qb_passing_yards_random_forest import fit_qb_passing_yards_random_forest, predict_qb_passing_yards_random_forest
from football.training.qb_passing_yards_xgboost import fit_qb_passing_yards_xgboost, predict_qb_passing_yards_xgboost
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


LIKELY_PRIMARY = "likely_primary"
NOT_LIKELY_PRIMARY = "not_likely_primary"
UNKNOWN = "unknown"
POPULATIONS = (LIKELY_PRIMARY, NOT_LIKELY_PRIMARY, UNKNOWN)


@dataclass(frozen=True)
class PregameSignalInventory:
    """One inspected candidate signal and its point-in-time suitability."""

    signal: str
    source: str
    keys: tuple[str, ...]
    historical_coverage: str
    update_timing: str
    known_before_kickoff: bool
    identifies_nonparticipants: bool
    conclusion: str


@dataclass(frozen=True)
class QBPrimaryFeasibilityReport:
    """Immutable audit of available primary-QB signals."""

    selector_supported_for_dated_snapshots: bool
    conclusion: str
    signals: tuple[PregameSignalInventory, ...]


@dataclass(frozen=True)
class LikelyPrimaryPopulationMetrics:
    """Metrics for one validation population; signed error is prediction minus actual."""

    population: str
    row_count: int
    metrics: RegressionMetrics | None
    mean_signed_error: float | None
    median_absolute_error: float | None
    percentile_75_absolute_error: float | None
    percentile_90_absolute_error: float | None
    mae_difference_from_full: float | None


@dataclass(frozen=True)
class LikelyPrimaryModelDiagnostics:
    """Full and population-specific validation diagnostics for one model."""

    model_name: str
    full_population: LikelyPrimaryPopulationMetrics
    populations: tuple[LikelyPrimaryPopulationMetrics, ...]


@dataclass(frozen=True)
class PopulationWeekCount:
    """Observed validation-row category counts for one season/week."""

    season: int
    week: int
    likely_primary_rows: int
    not_likely_primary_rows: int
    unknown_rows: int


@dataclass(frozen=True)
class TeamGameSelectionSummary:
    """Candidate selection coverage across scheduled validation team-games."""

    total_team_games: int
    zero_selected_team_games: int
    one_selected_team_games: int
    multiple_selected_team_games: int
    coverage_rate: float
    observed_unknown_row_count: int
    observed_unknown_rate: float


@dataclass(frozen=True)
class PopulationRowWinCount:
    """Rows where a model ties for smallest absolute error inside a population."""

    population: str
    model_name: str
    win_count: int


@dataclass(frozen=True)
class QBPrimaryPopulationAnalysis:
    """Immutable train/validation-only likely-primary population analysis."""

    feasibility: QBPrimaryFeasibilityReport
    selection_summary: TeamGameSelectionSummary
    population_counts_by_week: tuple[PopulationWeekCount, ...]
    model_diagnostics: tuple[LikelyPrimaryModelDiagnostics, ...]
    row_level_win_counts: tuple[PopulationRowWinCount, ...]


def audit_qb_primary_qb_feasibility() -> QBPrimaryFeasibilityReport:
    """Describe source suitability without loading data or making a selection."""

    signals = (
        PregameSignalInventory("weekly player statistics", "nflverse player stats / repository loader", ("season", "week", "game_id", "player_id"), "2020-2026", "postgame weekly summary", False, False, "Target-game participation and statistics are prohibited."),
        PregameSignalInventory("completed-game primary QB", "football.features.primary_quarterbacks", ("season", "week", "game_id", "team"), "completed QB games", "postgame attempt ranking", False, False, "Uses target-game attempts and is prohibited."),
        PregameSignalInventory("weekly roster", "nflreadpy.load_rosters_weekly", ("season", "week", "team", "player_id"), "2025 inspected; source supports historical weeks", "weekly status without snapshot timestamp", False, True, "Cannot prove availability before kickoff."),
        PregameSignalInventory("legacy depth chart", "nflverse depth charts", ("season", "week", "team", "player_id"), "2020-2024", "weekly row without timestamp", False, True, "Cannot prove a pre-kickoff snapshot."),
        PregameSignalInventory("dated depth chart", "nflverse depth charts dt snapshots", ("snapshot_timestamp", "team", "player_id", "depth_rank"), "2025-2026", "dated snapshot; only dates strictly before game date accepted", True, True, "Supports a conservative selector where snapshots and schedules are injected."),
        PregameSignalInventory("official starter", "repository/public loader inventory", (), "not available", "no timestamped official-starter field", False, False, "No usable source is loaded."),
        PregameSignalInventory("prior-game team and attempt history", "point-in-time training features", ("player_id", "team", "season", "week"), "2020-2026 histories", "strictly prior games", True, False, "Useful descriptive history but cannot enumerate nonparticipants."),
    )
    return QBPrimaryFeasibilityReport(True, "A deployable candidate selector is supported only by dated snapshots strictly before the scheduled game date; legacy weekly feeds are insufficient.", signals)


def analyze_qb_likely_primary_population(
    dataset_split: QBPassingYardsDatasetSplit,
    schedule_rows: pd.DataFrame,
    depth_chart_snapshots: pd.DataFrame,
) -> QBPrimaryPopulationAnalysis:
    """Fit five models on train and analyze validation-only pregame candidates.

    ``dataset_split.test`` is deliberately never read, copied, validated, or
    transformed. ``schedule_rows`` and ``depth_chart_snapshots`` are injected
    pregame inputs; dated snapshots on the target game date are excluded.
    """

    if not isinstance(dataset_split, QBPassingYardsDatasetSplit):
        raise TypeError("dataset_split must be a QBPassingYardsDatasetSplit")
    validation = _canonical_validation(dataset_split.validation)
    categories, summary = _classify_validation(validation, schedule_rows, depth_chart_snapshots)
    predictions = _fit_and_predict(dataset_split.train, validation)
    actual = _finite(validation[TARGET_COLUMN], "validation target_passing_yards").to_numpy()
    diagnostics = tuple(_model_diagnostics(name, validation, values, actual, categories) for name, values in predictions.items())
    return QBPrimaryPopulationAnalysis(
        feasibility=audit_qb_primary_qb_feasibility(),
        selection_summary=summary,
        population_counts_by_week=_week_counts(validation, categories),
        model_diagnostics=diagnostics,
        row_level_win_counts=_row_wins(predictions, actual, categories),
    )


def _canonical_validation(frame: pd.DataFrame) -> pd.DataFrame:
    required = [*TARGET_KEY, TARGET_COLUMN, "team"]
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Validation frame must be a pandas DataFrame")
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"Validation frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Validation frame must not be empty")
    if frame[TARGET_KEY].isna().any().any() or frame.duplicated(TARGET_KEY).any():
        raise ValueError("Validation frame must have unique, nonmissing target keys")
    _finite(frame[TARGET_COLUMN], "validation target_passing_yards")
    return frame.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True).copy(deep=True)


def _classify_validation(validation: pd.DataFrame, schedules: pd.DataFrame, snapshots: pd.DataFrame) -> tuple[np.ndarray, TeamGameSelectionSummary]:
    schedule = _schedule_context(schedules, validation)
    selected = _select_candidates(schedule, snapshots)
    row_context = validation.loc[:, [*TARGET_KEY, "team"]].merge(selected, on=["season", "week", "game_id", "team"], how="left", validate="many_to_one")
    categories = np.where(row_context["selected_player_id"].isna(), UNKNOWN, np.where(row_context["player_id"].eq(row_context["selected_player_id"]), LIKELY_PRIMARY, NOT_LIKELY_PRIMARY))
    total = len(schedule)
    one = len(selected)
    unknown = int((categories == UNKNOWN).sum())
    return categories.astype(object), TeamGameSelectionSummary(total, total - one, one, 0, one / total if total else 0.0, unknown, unknown / len(validation))


def _schedule_context(schedules: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    required = ["season", "week", "game_id", "team", "game_date"]
    if not isinstance(schedules, pd.DataFrame):
        raise TypeError("schedule_rows must be a pandas DataFrame")
    missing = sorted(set(required).difference(schedules.columns))
    if missing:
        raise ValueError(f"Schedule rows are missing required columns: {missing}")
    pairs = validation.loc[:, ["season", "week"]].drop_duplicates()
    data = schedules.merge(pairs, on=["season", "week"], how="inner")
    data = data.loc[data["game_id"].isin(validation["game_id"])].loc[:, required].copy()
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce").dt.date
    if data[required[:-1]].isna().any().any() or data["game_date"].isna().any():
        raise ValueError("Schedule rows have missing or invalid pregame context")
    if data.duplicated(required[:-1]).any():
        raise ValueError("Schedule rows have duplicate team-game keys")
    return data.sort_values(required[:-1], kind="mergesort").reset_index(drop=True)


def _select_candidates(schedule: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    required = ["team", "player_id", "position", "snapshot_timestamp", "depth_rank"]
    if not isinstance(snapshots, pd.DataFrame):
        raise TypeError("depth_chart_snapshots must be a pandas DataFrame")
    missing = sorted(set(required).difference(snapshots.columns))
    if missing:
        raise ValueError(f"Depth-chart snapshots are missing required columns: {missing}")
    data = snapshots.loc[:, required].copy()
    data["snapshot_timestamp"] = pd.to_datetime(data["snapshot_timestamp"], errors="coerce", utc=True)
    data["depth_rank"] = pd.to_numeric(data["depth_rank"], errors="coerce")
    valid = data["position"].eq("QB") & data["player_id"].notna() & data["team"].notna() & data["snapshot_timestamp"].notna() & data["depth_rank"].notna() & np.isfinite(data["depth_rank"]) & data["depth_rank"].ge(1) & data["depth_rank"].mod(1).eq(0)
    data = data.loc[valid].copy()
    if data.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", "selected_player_id"])
    data["snapshot_date"] = data["snapshot_timestamp"].dt.date
    joined = schedule.merge(data, on="team", how="left")
    joined = joined.loc[(joined["snapshot_date"] < joined["game_date"]) & (joined["snapshot_date"].map(lambda value: value.year) == joined["season"])].copy()
    if joined.empty:
        return pd.DataFrame(columns=["season", "week", "game_id", "team", "selected_player_id"])
    latest = joined.groupby(["season", "week", "game_id", "team"], sort=False)["snapshot_date"].transform("max")
    candidates = joined.loc[joined["snapshot_date"].eq(latest)].copy()
    best = candidates.groupby(["season", "week", "game_id", "team"], sort=False)["depth_rank"].transform("min")
    candidates = candidates.loc[candidates["depth_rank"].eq(best)].copy()
    candidates["_id"] = candidates["player_id"].astype(str)
    return candidates.sort_values(["season", "week", "game_id", "team", "_id"], kind="mergesort").drop_duplicates(["season", "week", "game_id", "team"]).rename(columns={"player_id": "selected_player_id"}).loc[:, ["season", "week", "game_id", "team", "selected_player_id"]].reset_index(drop=True)


def _fit_and_predict(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, np.ndarray]:
    fitted = (
        ("Historical average", fit_qb_passing_yards_baseline(train), predict_qb_passing_yards_baseline),
        ("Linear Regression", fit_qb_passing_yards_linear_regression(train), predict_qb_passing_yards_linear_regression),
        ("Random Forest", fit_qb_passing_yards_random_forest(train), predict_qb_passing_yards_random_forest),
        ("sklearn Gradient Boosting", fit_qb_passing_yards_gradient_boosting(train), predict_qb_passing_yards_gradient_boosting),
        ("XGBoost", fit_qb_passing_yards_xgboost(train), predict_qb_passing_yards_xgboost),
    )
    result = {}
    for name, model, predict in fitted:
        output = predict(model, validation)
        columns = [column for column in output.columns if column.endswith("_prediction")]
        if len(columns) != 1 or output.duplicated(TARGET_KEY).any():
            raise ValueError(f"{name} prediction output is structurally invalid")
        if set(map(tuple, output[TARGET_KEY].to_numpy())) != set(map(tuple, validation[TARGET_KEY].to_numpy())):
            raise ValueError(f"{name} prediction keys do not match validation keys")
        result[name] = _finite(output.sort_values(TARGET_KEY, kind="mergesort")[columns[0]], f"{name} predictions").to_numpy()
    if tuple(result) != MODEL_NAMES:
        raise ValueError("Completed model contract changed")
    return result


def _model_diagnostics(name: str, validation: pd.DataFrame, predictions: np.ndarray, actual: np.ndarray, categories: np.ndarray) -> LikelyPrimaryModelDiagnostics:
    full = _population_metrics("full_population", validation, predictions, actual, np.ones(len(validation), dtype=bool), None)
    populations = tuple(_population_metrics(category, validation, predictions, actual, categories == category, full.metrics.mae) for category in POPULATIONS)
    return LikelyPrimaryModelDiagnostics(name, full, populations)


def _population_metrics(name: str, validation: pd.DataFrame, predictions: np.ndarray, actual: np.ndarray, mask: np.ndarray, full_mae: float | None) -> LikelyPrimaryPopulationMetrics:
    count = int(mask.sum())
    if count == 0:
        return LikelyPrimaryPopulationMetrics(name, 0, None, None, None, None, None, None)
    selected = validation.loc[mask]
    values = predictions[mask]
    output = selected.loc[:, TARGET_KEY].copy()
    output[PREDICTION_COLUMN] = values
    metrics = evaluate_qb_passing_yards_predictions(selected, output)
    errors = values - actual[mask]
    return LikelyPrimaryPopulationMetrics(name, count, metrics, float(errors.mean()), float(np.percentile(np.abs(errors), 50, method=PERCENTILE_METHOD)), float(np.percentile(np.abs(errors), 75, method=PERCENTILE_METHOD)), float(np.percentile(np.abs(errors), 90, method=PERCENTILE_METHOD)), None if full_mae is None else float(metrics.mae - full_mae))


def _week_counts(validation: pd.DataFrame, categories: np.ndarray) -> tuple[PopulationWeekCount, ...]:
    rows = []
    for (season, week), indices in validation.groupby(["season", "week"], sort=True).groups.items():
        values = categories[np.asarray(list(indices), dtype=int)]
        rows.append(PopulationWeekCount(int(season), int(week), int((values == LIKELY_PRIMARY).sum()), int((values == NOT_LIKELY_PRIMARY).sum()), int((values == UNKNOWN).sum())))
    return tuple(rows)


def _row_wins(predictions: dict[str, np.ndarray], actual: np.ndarray, categories: np.ndarray) -> tuple[PopulationRowWinCount, ...]:
    matrix = np.vstack(tuple(predictions.values()))
    errors = np.abs(matrix - actual)
    results = []
    for category in POPULATIONS:
        mask = categories == category
        if not mask.any():
            results.extend(PopulationRowWinCount(category, name, 0) for name in predictions)
            continue
        minimums = errors[:, mask].min(axis=0)
        results.extend(PopulationRowWinCount(category, name, int(np.isclose(errors[index, mask], minimums, rtol=0.0, atol=PREDICTION_TIE_TOLERANCE).sum())) for index, name in enumerate(predictions))
    return tuple(results)


def _finite(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(f"{label} values must be finite")
    return numeric.astype(float)
