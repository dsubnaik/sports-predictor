"""Load and normalize nflverse football data via nflreadpy."""

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from football.data.build_quarterback_dataset import build_quarterback_dataset


PLAYER_STATS_SOURCE_COLUMNS = [
    "season",
    "week",
    "season_type",
    "game_id",
    "player_id",
    "player_display_name",
    "position",
    "team",
    "opponent_team",
    "attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
]

PLAYER_STATS_CONTRACT_MAPPING = {
    "season": "season",
    "week": "week",
    "game_id": "game_id",
    "player_id": "player_id",
    "player_display_name": "player_name",
    "team": "team",
    "opponent_team": "opponent",
    "attempts": "passing_attempts",
    "completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_touchdowns",
    "passing_interceptions": "interceptions",
}


def _to_pandas(data: Any) -> pd.DataFrame:
    """Convert nflreadpy's Polars output into pandas at the source boundary."""

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    raise TypeError(
        "Expected a pandas DataFrame or an object with a to_pandas() method."
    )


def _load_nflreadpy() -> Any:
    import nflreadpy

    return nflreadpy


def load_player_game_stats(
    seasons: int | Sequence[int] | None,
    loader: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Load player game-level statistics for the requested NFL seasons."""

    if loader is None:
        loader = _load_nflreadpy().load_player_stats

    source_data = loader(seasons=seasons, summary_level="week")
    return _to_pandas(source_data)


def load_schedules(
    seasons: int | Sequence[int] | None,
    loader: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """Load NFL schedules for the requested seasons."""

    if loader is None:
        loader = _load_nflreadpy().load_schedules

    source_data = loader(seasons=seasons)
    return _to_pandas(source_data)


def validate_player_stats_schema(data: pd.DataFrame) -> None:
    """Validate source columns needed to normalize quarterback game rows."""

    missing_columns = sorted(
        set(PLAYER_STATS_SOURCE_COLUMNS).difference(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "NFL player stats source data is missing required columns: "
            f"{missing_columns}"
        )


def normalize_quarterback_game_stats(
    player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Map nflverse player stats into the quarterback dataset contract."""

    validate_player_stats_schema(player_stats)

    quarterback_rows = player_stats.loc[
        (player_stats["position"] == "QB")
        & (player_stats["season_type"] == "REG"),
        PLAYER_STATS_CONTRACT_MAPPING.keys(),
    ].rename(columns=PLAYER_STATS_CONTRACT_MAPPING)

    return build_quarterback_dataset(quarterback_rows)
