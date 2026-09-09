"""Deterministically attach nflverse identities to normalized NFL prop odds."""

from __future__ import annotations

import math
import unicodedata
from numbers import Real
from typing import Any

import pandas as pd

from football.odds.player_props import (
    PLAYER_PROP_ODDS_COLUMNS,
    SUPPORTED_PLAYER_PROP_MARKETS,
)


PLAYER_PROP_MATCH_COLUMNS = [
    "player_id",
    "nflverse_player_name",
    "nflverse_team",
    "expected_position",
    "match_status",
    "match_method",
    "match_candidate_count",
    "match_note",
]
PLAYER_PROP_MATCH_OUTPUT_COLUMNS = PLAYER_PROP_ODDS_COLUMNS + PLAYER_PROP_MATCH_COLUMNS

_MARKET_POSITIONS = {
    "player_pass_yds": "QB",
    "player_rush_yds": "RB",
}
_PLAYER_REFERENCE_COLUMNS = ["player_id", "player_name", "team", "position"]
_MATCH_METHOD = "canonical_name_and_position"


def match_player_prop_odds(odds: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach an nflverse ID when canonical name and position are unique.

    ``players`` is a current/reference view, not a historical roster: rows for
    one player ID must agree on display name, team, and position. Exact or
    repeated agreeing snapshots collapse. QB/RB rows without an ID are treated
    as unresolved reference records and cannot become candidates.
    """

    _validate_odds(odds)
    candidates = _validated_candidates(players)
    if odds.empty:
        return pd.DataFrame(columns=PLAYER_PROP_MATCH_OUTPUT_COLUMNS)

    candidate_lookup = _candidate_lookup(candidates)
    result = odds.loc[:, PLAYER_PROP_ODDS_COLUMNS].copy().reset_index(drop=True)
    match_rows = []
    for _, row in result.iterrows():
        expected_position = _MARKET_POSITIONS[row["market_key"]]
        canonical_name = _canonical_name(row["player_name"], "odds player_name")
        choices = candidate_lookup.get((canonical_name, expected_position), [])
        match_rows.append(
            _match_values(choices, expected_position)
        )

    matches = pd.DataFrame(match_rows, columns=PLAYER_PROP_MATCH_COLUMNS)
    return pd.concat([result, matches], axis=1).reset_index(drop=True)


def _validate_odds(odds: pd.DataFrame) -> None:
    if not isinstance(odds, pd.DataFrame):
        raise TypeError("odds must be a pandas DataFrame")
    _reject_duplicate_columns(odds, "Odds data")
    if odds.columns.tolist() != PLAYER_PROP_ODDS_COLUMNS:
        raise ValueError(
            "Odds data columns must exactly match PLAYER_PROP_ODDS_COLUMNS "
            "in order"
        )
    for index, value in odds["market_key"].items():
        if value not in SUPPORTED_PLAYER_PROP_MARKETS:
            raise ValueError(f"odds market_key at index {index} is unsupported")
    for index, value in odds["player_name"].items():
        _canonical_name(value, f"odds player_name at index {index}")
    for column in ["price", "point"]:
        for index, value in odds[column].items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"odds {column} at index {index} must be a finite numeric "
                    "value and not a boolean"
                )


def _validated_candidates(players: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(players, pd.DataFrame):
        raise TypeError("players must be a pandas DataFrame")
    _reject_duplicate_columns(players, "Player reference data")
    missing = sorted(set(_PLAYER_REFERENCE_COLUMNS).difference(players.columns))
    if missing:
        raise ValueError(f"Player reference data is missing required columns: {missing}")

    reference = players.loc[:, _PLAYER_REFERENCE_COLUMNS].copy()
    _validate_reference_text(reference, "team")
    _validate_reference_text(reference, "position")
    reference["_position"] = reference["position"].str.strip().str.upper()
    relevant = reference.loc[reference["_position"].isin({"QB", "RB"})].copy()
    if relevant.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "player_name",
                "team",
                "position",
                "_canonical_name",
                "_position",
            ]
        )

    valid_id = relevant["player_id"].map(_valid_identity_text)
    invalid_id = ~valid_id & ~relevant["player_id"].map(_missing_value)
    if invalid_id.any():
        raise ValueError("QB/RB player-reference player_id values must be nonblank strings or missing")
    candidates = relevant.loc[valid_id].copy()
    candidates["player_id"] = candidates["player_id"].str.strip()
    for index, value in candidates["player_name"].items():
        _canonical_name(value, f"player-reference player_name at index {index}")
    candidates["_canonical_name"] = candidates["player_name"].map(
        lambda value: _canonical_name(value, "player-reference player_name")
    )
    _reject_conflicting_player_identities(candidates)
    return candidates.drop_duplicates(
        subset=["player_id", "player_name", "team", "position", "_canonical_name"]
    ).reset_index(drop=True)


def _candidate_lookup(candidates: pd.DataFrame) -> dict[tuple[str, str], list[dict[str, str]]]:
    lookup: dict[tuple[str, str], list[dict[str, str]]] = {}
    for key, group in candidates.groupby(["_canonical_name", "_position"], sort=True):
        choices = (
            group.loc[:, ["player_id", "player_name", "team"]]
            .drop_duplicates(subset=["player_id"])
            .sort_values("player_id", kind="mergesort")
            .to_dict("records")
        )
        lookup[key] = choices
    return lookup


def _match_values(choices: list[dict[str, str]], expected_position: str) -> dict[str, Any]:
    count = len(choices)
    if count == 1:
        candidate = choices[0]
        return {
            "player_id": candidate["player_id"],
            "nflverse_player_name": candidate["player_name"],
            "nflverse_team": candidate["team"],
            "expected_position": expected_position,
            "match_status": "matched",
            "match_method": _MATCH_METHOD,
            "match_candidate_count": 1,
            "match_note": "Unique canonical name-and-position candidate",
        }
    if count == 0:
        return {
            "player_id": pd.NA,
            "nflverse_player_name": pd.NA,
            "nflverse_team": pd.NA,
            "expected_position": expected_position,
            "match_status": "unmatched",
            "match_method": pd.NA,
            "match_candidate_count": 0,
            "match_note": "No canonical name-and-position candidate",
        }
    return {
        "player_id": pd.NA,
        "nflverse_player_name": pd.NA,
        "nflverse_team": pd.NA,
        "expected_position": expected_position,
        "match_status": "ambiguous",
        "match_method": pd.NA,
        "match_candidate_count": count,
        "match_note": "Multiple canonical name-and-position candidates",
    }


def _reject_conflicting_player_identities(candidates: pd.DataFrame) -> None:
    conflicts = []
    for player_id, group in candidates.groupby("player_id", sort=True):
        identities = group.loc[:, ["player_name", "team", "position"]].drop_duplicates()
        if len(identities) > 1:
            conflicts.append(player_id)
    if conflicts:
        raise ValueError(
            "Conflicting player-reference identities for player_id values: "
            f"{sorted(conflicts)}"
        )


def _canonical_name(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be nonblank text")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    for punctuation in ".',":
        normalized = normalized.replace(punctuation, "")
    tokens = normalized.split()
    initials = []
    while tokens and len(tokens[0]) == 1:
        initials.append(tokens.pop(0))
    if initials:
        tokens.insert(0, "".join(initials))
    normalized = " ".join(tokens)
    if not normalized:
        raise ValueError(f"{field} must be nonblank text")
    return normalized


def _validate_reference_text(reference: pd.DataFrame, column: str) -> None:
    for index, value in reference[column].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"player-reference {column} at index {index} must be nonblank text"
            )


def _valid_identity_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _missing_value(value: Any) -> bool:
    return value is None or value is pd.NA or (
        isinstance(value, float) and math.isnan(value)
    )


def _reject_duplicate_columns(data: pd.DataFrame, label: str) -> None:
    duplicates = data.columns[data.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{label} contains duplicate columns: {duplicates}")
