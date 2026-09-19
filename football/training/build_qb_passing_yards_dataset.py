"""Build a leakage-safe raw dataset for QB passing-yards modeling.

Each row is one completed, regular-season quarterback-game.  Audit metadata
identifies the target game and player; only :data:`FEATURE_COLUMNS` are
predictive inputs.  ``target_passing_yards`` is the target game's actual
passing yards and is never used as a feature.

Week 1 histories use only the preceding regular season.  Week 2 and later
histories use only games in the target season with a strictly smaller week.
Consequently, the target game and every other game in its week are excluded,
regardless of kickoff time.  Missing history remains missing (with zero sample
sizes and an indicator); this raw builder deliberately performs no imputation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football.data.build_quarterback_dataset import build_quarterback_dataset
from football.features.defense_vs_quarterbacks import (
    build_defense_vs_primary_quarterback_logs,
)
from football.features.primary_quarterbacks import identify_primary_quarterbacks


TARGET_KEY = ["season", "week", "game_id", "player_id"]

METADATA_COLUMNS = [
    "season",
    "week",
    "game_id",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "home_away",
]

TARGET_COLUMN = "target_passing_yards"

FEATURE_COLUMNS = [
    "qb_season_passing_yards_avg",
    "qb_last3_passing_yards_avg",
    "qb_season_passing_attempts_avg",
    "qb_last3_passing_attempts_avg",
    "qb_season_history_games",
    "qb_last3_history_games",
    "qb_missing_history",
    "defense_season_passing_yards_allowed_avg",
    "defense_last3_passing_yards_allowed_avg",
    "defense_season_passing_attempts_allowed_avg",
    "defense_season_history_games",
    "defense_last3_history_games",
    "defense_missing_history",
    "defense_matchup_rank",
]

OUTPUT_COLUMNS = [*METADATA_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]

SCHEDULE_CONTEXT_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "home_away",
]
SCHEDULE_KEY = ["season", "week", "game_id", "team"]


def build_qb_passing_yards_training_dataset(
    quarterback_games: pd.DataFrame,
    schedule_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return one point-in-time row per eligible completed QB-game.

    ``quarterback_games`` must use the normalized QB-game contract and
    ``schedule_rows`` the normalized team-game schedule contract.  A target is
    eligible when its QB row has a unique, agreeing schedule perspective and a
    finite passing-yards value.  The schedule contract already represents only
    regular-season games, so unmapped QB rows (including postseason rows) are
    excluded rather than guessed.

    Defensive histories use the existing primary-QB definition for *prior*
    defense games only.  It is never used to select target rows, so multiple
    QBs from a target game remain separate observations.
    """

    normalized_qbs = build_quarterback_dataset(quarterback_games)
    normalized_schedule = _normalize_schedule_context(schedule_rows)
    contextual_qbs = _attach_schedule_context(normalized_qbs, normalized_schedule)

    targets = contextual_qbs.loc[
        _finite_numeric(contextual_qbs["passing_yards"])
    ].copy()
    if targets.empty:
        return _empty_output()

    # Coercion is confined to this owned working frame.  Invalid historical
    # values remain unavailable to aggregation rather than receiving a fill.
    history_qbs = contextual_qbs.copy()
    history_qbs["passing_yards"] = pd.to_numeric(
        history_qbs["passing_yards"], errors="coerce"
    ).where(lambda values: np.isfinite(values))
    history_qbs["passing_attempts"] = pd.to_numeric(
        history_qbs["passing_attempts"], errors="coerce"
    ).where(lambda values: np.isfinite(values))
    defense_logs = build_defense_vs_primary_quarterback_logs(
        identify_primary_quarterbacks(history_qbs)
    )

    rows = []
    for target in targets.sort_values(TARGET_KEY, kind="mergesort").itertuples(
        index=False
    ):
        history_season, history_week = _history_cutoff(target.season, target.week)
        qb_prior = _prior_rows(
            history_qbs, "player_id", target.player_id, history_season, history_week
        )
        defense_prior = _prior_rows(
            defense_logs, "defense", target.opponent, history_season, history_week
        )
        defense_window = defense_logs.loc[
            (defense_logs["season"] == history_season)
            & (defense_logs["week"] < history_week)
        ]

        row = {column: getattr(target, column) for column in METADATA_COLUMNS}
        row[TARGET_COLUMN] = float(target.passing_yards)
        row.update(_qb_features(qb_prior))
        row.update(_defense_features(defense_prior, defense_window, target.opponent))
        rows.append(row)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    result = result.sort_values(TARGET_KEY, kind="mergesort").reset_index(drop=True)
    _validate_unique_target_keys(result)
    return result


def _normalize_schedule_context(schedule_rows: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(SCHEDULE_CONTEXT_COLUMNS).difference(schedule_rows))
    if missing_columns:
        raise ValueError(
            "Normalized schedule data is missing required columns: "
            f"{missing_columns}"
        )

    context = schedule_rows.loc[:, SCHEDULE_CONTEXT_COLUMNS].drop_duplicates().copy()
    conflicts = context.loc[context.duplicated(SCHEDULE_KEY, keep=False)]
    if not conflicts.empty:
        keys = (
            conflicts.loc[:, SCHEDULE_KEY]
            .drop_duplicates()
            .sort_values(SCHEDULE_KEY, kind="mergesort")
            .to_dict(orient="records")
        )
        raise ValueError(f"Conflicting normalized schedule rows found for keys: {keys}")

    _validate_schedule_values(context)
    return context.sort_values(SCHEDULE_KEY, kind="mergesort").reset_index(drop=True)


def _validate_schedule_values(context: pd.DataFrame) -> None:
    invalid = context.loc[
        context[["team", "opponent", "home_away"]].isna().any(axis=1)
        | (context["team"].astype(str).str.strip() == "")
        | (context["opponent"].astype(str).str.strip() == "")
        | ~context["home_away"].isin(["home", "away"])
        | (context["team"] == context["opponent"])
    ]
    if not invalid.empty:
        keys = invalid.loc[:, SCHEDULE_KEY].to_dict(orient="records")
        raise ValueError(f"Normalized schedule rows have invalid context for keys: {keys}")


def _attach_schedule_context(
    quarterback_games: pd.DataFrame, schedule_context: pd.DataFrame
) -> pd.DataFrame:
    joined = quarterback_games.merge(
        schedule_context,
        on=SCHEDULE_KEY,
        how="left",
        validate="many_to_one",
        suffixes=("", "_schedule"),
        indicator=True,
    )
    missing = joined.loc[joined["_merge"] != "both", TARGET_KEY]
    if not missing.empty:
        keys = missing.sort_values(TARGET_KEY, kind="mergesort").to_dict(orient="records")
        raise ValueError(f"Quarterback games are missing schedule context for keys: {keys}")

    mismatch = joined.loc[joined["opponent"] != joined["opponent_schedule"], TARGET_KEY]
    if not mismatch.empty:
        keys = mismatch.sort_values(TARGET_KEY, kind="mergesort").to_dict(orient="records")
        raise ValueError(f"Quarterback games disagree with schedule opponents for keys: {keys}")

    return (
        joined.drop(columns=["opponent_schedule", "_merge"])
        .sort_values(TARGET_KEY, kind="mergesort")
        .reset_index(drop=True)
    )


def _history_cutoff(season: int, week: int) -> tuple[int, int]:
    return (season - 1, np.iinfo(np.int64).max) if week == 1 else (season, week)


def _prior_rows(
    data: pd.DataFrame,
    entity_column: str,
    entity: object,
    history_season: int,
    history_week: int,
) -> pd.DataFrame:
    return data.loc[
        (data[entity_column] == entity)
        & (data["season"] == history_season)
        & (data["week"] < history_week)
    ].sort_values(["season", "week", "game_id"], kind="mergesort")


def _qb_features(prior: pd.DataFrame) -> dict[str, object]:
    last3 = prior.tail(3)
    usable = prior.loc[_finite_numeric(prior["passing_yards"])]
    usable_last3 = last3.loc[_finite_numeric(last3["passing_yards"])]
    return {
        "qb_season_passing_yards_avg": usable["passing_yards"].mean(),
        "qb_last3_passing_yards_avg": usable_last3["passing_yards"].mean(),
        "qb_season_passing_attempts_avg": prior.loc[
            _finite_numeric(prior["passing_attempts"]), "passing_attempts"
        ].mean(),
        "qb_last3_passing_attempts_avg": last3.loc[
            _finite_numeric(last3["passing_attempts"]), "passing_attempts"
        ].mean(),
        "qb_season_history_games": len(usable),
        "qb_last3_history_games": len(usable_last3),
        "qb_missing_history": len(usable) == 0,
    }


def _defense_features(
    prior: pd.DataFrame, defense_window: pd.DataFrame, defense: object
) -> dict[str, object]:
    last3 = prior.tail(3)
    usable = prior.loc[_finite_numeric(prior["passing_yards_allowed"])]
    usable_last3 = last3.loc[_finite_numeric(last3["passing_yards_allowed"])]
    ranked = (
        defense_window.loc[
            _finite_numeric(defense_window["passing_yards_allowed"])
        ]
        .groupby("defense", sort=True)["passing_yards_allowed"]
        .mean()
        .rank(method="min", ascending=False)
    )
    return {
        "defense_season_passing_yards_allowed_avg": usable[
            "passing_yards_allowed"
        ].mean(),
        "defense_last3_passing_yards_allowed_avg": usable_last3[
            "passing_yards_allowed"
        ].mean(),
        "defense_season_passing_attempts_allowed_avg": prior.loc[
            _finite_numeric(prior["passing_attempts_allowed"]),
            "passing_attempts_allowed",
        ].mean(),
        "defense_season_history_games": len(usable),
        "defense_last3_history_games": len(usable_last3),
        "defense_missing_history": len(usable) == 0,
        "defense_matchup_rank": ranked.get(defense, np.nan),
    }


def _finite_numeric(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(np.isfinite(numeric), index=values.index)


def _validate_unique_target_keys(data: pd.DataFrame) -> None:
    if data.duplicated(TARGET_KEY).any():
        raise ValueError("QB passing-yards training dataset has duplicate target keys")


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)
