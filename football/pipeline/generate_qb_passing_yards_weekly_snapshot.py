"""Explicit, unscored generation of the current QB projection snapshot.

This module is deliberately a small composition layer. It neither retrains a
model during normal weekly generation nor recreates feature or snapshot logic.
The fitted joblib artifact is trusted-local only; manifest checks detect an
incompatible or changed artifact but do not make untrusted joblib safe.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from football.config import (
    QB_PASSING_YARDS_PRODUCTION_ARTIFACT_PATH,
    QB_PASSING_YARDS_PROJECTION_SNAPSHOT_PATH,
)
from football.data.build_schedule_dataset import normalize_schedule_dataset
from football.data.fetch_nflverse import (
    load_depth_charts,
    load_player_game_stats,
    load_schedules,
    normalize_quarterback_game_stats,
)
from football.pipeline.qb_passing_yards_weekly_inference import (
    generate_qb_passing_yards_weekly_projections,
)
from football.pipeline.qb_passing_yards_weekly_projection_snapshots import (
    QBPassingYardsWeeklyProjectionSnapshotArtifact,
    save_qb_passing_yards_weekly_projection_snapshot,
)
from football.training.build_qb_passing_yards_dataset import (
    TARGET_KEY,
    build_qb_passing_yards_training_dataset,
)
from football.training.qb_passing_yards_production_model import (
    load_qb_passing_yards_production_model,
    save_qb_passing_yards_production_model,
    train_qb_passing_yards_production_model,
)
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit


SOURCE_TRAINING_SEASONS = (2020, 2021, 2022, 2023, 2024, 2025)
MODELING_TRAINING_SEASONS = (2021, 2022, 2023, 2024, 2025)
VALIDATION_START = (2025, 1)
HOLDOUT_BOUNDARY = (2026, 1)


@dataclass(frozen=True)
class QBWeeklySnapshotGenerationResult:
    """Aggregate-only result of one dry run or explicit snapshot write."""

    season: int
    week: int
    dry_run: bool
    snapshot_written: bool
    snapshot_path: str
    artifact_training_rows: int
    artifact_maximum_training_period: tuple[int, int]
    artifact_checksum_prefix: str
    scheduled_games_considered: int
    team_games_considered: int
    selected_candidates: int
    projections: int
    unknown_team_games: int
    cold_starts: int
    defense_missing: int
    skipped_reasons: tuple[tuple[str, int], ...]


def prepare_qb_passing_yards_production_artifact(
    *,
    artifact_path: str | Path = QB_PASSING_YARDS_PRODUCTION_ARTIFACT_PATH,
    overwrite: bool = False,
    player_loader: Callable | None = None,
    schedule_loader: Callable | None = None,
):
    """Explicitly refit the frozen candidate on 2021--2025 targets only.

    The standard chronological splitter requires a nonempty held-out partition.
    This operation intentionally loads no 2026 source rows, so it constructs
    the established 2021--2024 / 2025 split directly.
    """
    qbs = normalize_quarterback_game_stats(
        load_player_game_stats(list(SOURCE_TRAINING_SEASONS), loader=player_loader)
    )
    schedules = normalize_schedule_dataset(
        load_schedules(list(SOURCE_TRAINING_SEASONS), loader=schedule_loader)
    )
    dataset = build_qb_passing_yards_training_dataset(qbs, schedules)
    data = dataset.loc[dataset["season"].isin(MODELING_TRAINING_SEASONS)].copy()
    train = _canonical_period_copy(data, lambda period: period < VALIDATION_START)
    validation = _canonical_period_copy(
        data, lambda period: VALIDATION_START <= period < HOLDOUT_BOUNDARY
    )
    if train.empty or validation.empty:
        raise ValueError(
            "Artifact preparation requires nonempty 2021--2024 train and 2025 validation rows"
        )
    # The production fitter deliberately never reads ``test``. An empty frame
    # makes that boundary explicit without requesting 2026 source data.
    split = QBPassingYardsDatasetSplit(
        train=train, validation=validation, test=data.iloc[:0].copy()
    )
    model = train_qb_passing_yards_production_model(split)
    return save_qb_passing_yards_production_model(model, artifact_path, overwrite=overwrite)


def generate_qb_passing_yards_weekly_snapshot(
    *,
    season: int,
    week: int,
    as_of_utc: object,
    artifact_path: str | Path = QB_PASSING_YARDS_PRODUCTION_ARTIFACT_PATH,
    snapshot_path: str | Path = QB_PASSING_YARDS_PROJECTION_SNAPSHOT_PATH,
    dry_run: bool = False,
    overwrite_snapshot: bool = False,
    player_loader: Callable | None = None,
    schedule_loader: Callable | None = None,
    depth_loader: Callable | None = None,
) -> QBWeeklySnapshotGenerationResult:
    """Generate an unscored snapshot from a validated, already-fitted artifact."""
    _validate_requested_period(season, week)
    timestamp = _utc_timestamp(as_of_utc, "as_of_utc")
    loaded = load_qb_passing_yards_production_model(artifact_path)
    if loaded.metadata.maximum_training_period > (2025, 18):
        raise ValueError("Production artifact training boundary is incompatible with live inference")

    history_seasons = list(range(2020, season + 1))
    qbs = normalize_quarterback_game_stats(
        load_player_game_stats(history_seasons, loader=player_loader)
    )
    # Any delivered target/future source rows are removed before feature
    # construction. The inference API independently enforces its weekly cutoff.
    qbs = qbs.loc[_before_period(qbs, season, week)].copy()
    schedules = _inference_schedule(
        load_schedules(history_seasons, loader=schedule_loader)
    )
    depth = _inference_depth(load_depth_charts([season], loader=depth_loader))
    report = generate_qb_passing_yards_weekly_projections(
        loaded,
        qbs,
        schedules,
        depth,
        target_season=season,
        target_week=week,
        as_of_utc=timestamp,
    )
    if not report.projections:
        raise ValueError("No future-eligible QB projections were generated")

    digest = _sha256_file(Path(artifact_path))
    diagnostics = report.feature_diagnostics
    if not dry_run:
        artifact = QBPassingYardsWeeklyProjectionSnapshotArtifact(
            loaded.artifact_name, digest, loaded.metadata
        )
        save_qb_passing_yards_weekly_projection_snapshot(
            report, snapshot_path, artifact, overwrite=overwrite_snapshot
        )
    reasons: dict[str, int] = {}
    for skip in diagnostics.skips:
        reasons[skip.reason] = reasons.get(skip.reason, 0) + 1
    return QBWeeklySnapshotGenerationResult(
        season=season,
        week=week,
        dry_run=dry_run,
        snapshot_written=not dry_run,
        snapshot_path=str(snapshot_path),
        artifact_training_rows=loaded.metadata.training_row_count,
        artifact_maximum_training_period=loaded.metadata.maximum_training_period,
        artifact_checksum_prefix=digest[:12],
        scheduled_games_considered=diagnostics.scheduled_games_considered,
        team_games_considered=diagnostics.team_games_considered,
        selected_candidates=diagnostics.likely_primary_selections,
        projections=len(report.projections),
        unknown_team_games=diagnostics.unknown_team_games,
        cold_starts=diagnostics.cold_start_projections,
        defense_missing=diagnostics.defense_missing_projections,
        skipped_reasons=tuple(sorted(reasons.items())),
    )


def _canonical_period_copy(
    data: pd.DataFrame, include: Callable[[tuple[int, int]], bool]
) -> pd.DataFrame:
    periods = list(zip(data["season"], data["week"], strict=True))
    return (
        data.loc[[include((int(season), int(week))) for season, week in periods]]
        .sort_values(TARGET_KEY, kind="mergesort")
        .reset_index(drop=True)
        .copy(deep=True)
    )


def _before_period(data: pd.DataFrame, season: int, week: int) -> pd.Series:
    return (data["season"] < season) | (
        (data["season"] == season) & (data["week"] < week)
    )


def _inference_schedule(raw: pd.DataFrame) -> pd.DataFrame:
    """Attach timezone-aware kickoff timestamps to normalized schedule rows.

    nflverse ``gameday`` / ``gametime`` is interpreted as Eastern local
    schedule time. Invalid clocks remain missing and become explicit skips in
    weekly inference rather than being silently treated as UTC.
    """
    data = normalize_schedule_dataset(raw)
    clocks = pd.to_datetime(
        raw["gameday"].astype(str) + " " + raw["gametime"].astype(str),
        errors="coerce",
    )
    kickoff = clocks.dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="NaT"
    ).dt.tz_convert("UTC")
    mapping = dict(zip(raw["game_id"], kickoff, strict=True))
    data["scheduled_kickoff"] = data["game_id"].map(mapping)
    data["season_type"] = "REG"
    return data


def _inference_depth(raw: pd.DataFrame) -> pd.DataFrame:
    required = ["dt", "team", "gsis_id", "pos_abb", "pos_rank"]
    missing = sorted(set(required).difference(raw.columns))
    if missing:
        raise ValueError(f"Depth charts are missing required columns: {missing}")
    columns = [*required, *(["player_name"] if "player_name" in raw else [])]
    return raw.loc[:, columns].rename(
        columns={
            "dt": "snapshot_timestamp",
            "gsis_id": "player_id",
            "pos_abb": "position",
            "pos_rank": "depth_rank",
        }
    ).copy()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _validate_requested_period(season: object, week: object) -> None:
    if (
        isinstance(season, bool)
        or isinstance(week, bool)
        or not isinstance(season, int)
        or not isinstance(week, int)
    ):
        raise ValueError("season and week must be integer values")
    if season < 2026 or not 1 <= week <= 18:
        raise ValueError(
            "season must be 2026 or later and week must be a regular-season week from 1 through 18"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an unscored weekly QB projection snapshot.")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--as-of")
    parser.add_argument("--artifact-path", default=str(QB_PASSING_YARDS_PRODUCTION_ARTIFACT_PATH))
    parser.add_argument("--snapshot-path", default=str(QB_PASSING_YARDS_PROJECTION_SNAPSHOT_PATH))
    parser.add_argument("--prepare-artifact", action="store_true")
    parser.add_argument("--overwrite-artifact", action="store_true")
    parser.add_argument("--overwrite-snapshot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.overwrite_artifact and not args.prepare_artifact:
        parser.error("--overwrite-artifact requires --prepare-artifact")
    if args.prepare_artifact:
        prepare_qb_passing_yards_production_artifact(
            artifact_path=args.artifact_path, overwrite=args.overwrite_artifact
        )
    requested = (args.season, args.week, args.as_of)
    if all(value is None for value in requested):
        if args.prepare_artifact:
            print("Trusted production artifact prepared; no weekly snapshot requested.")
            return 0
        parser.error("--season, --week, and --as-of are required for weekly generation")
    if any(value is None for value in requested):
        parser.error("--season, --week, and --as-of must be supplied together")
    try:
        result = generate_qb_passing_yards_weekly_snapshot(
            season=args.season,
            week=args.week,
            as_of_utc=args.as_of,
            artifact_path=args.artifact_path,
            snapshot_path=args.snapshot_path,
            dry_run=args.dry_run,
            overwrite_snapshot=args.overwrite_snapshot,
        )
    except FileNotFoundError as error:
        parser.error(f"{error}. Run the explicit artifact preparation command first.")
    print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
