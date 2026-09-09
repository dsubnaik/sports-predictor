"""Unit tests for append-only pending football decision recording."""

import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import football.decisions as decisions
from football.decisions import (
    DecisionIdConflictError,
    DecisionValidationError,
    DuplicateDecisionError,
    PendingDecision,
    get_decision,
    initialize_decision_store,
    list_decisions,
    record_decision,
)


ODDS_TIME = datetime(2026, 9, 10, 17, 0, tzinfo=timezone.utc)
RECORDED_TIME = datetime(2026, 9, 10, 17, 5, tzinfo=timezone.utc)


@pytest.fixture
def connection():
    database = sqlite3.connect(":memory:")
    initialize_decision_store(database)
    yield database
    database.close()


def qb_decision(**changes):
    values = dict(
        decision_id="decision-qb", position=" QB ", player_id=" qb-1 ", player_name=" Alex QB ",
        team=" KC ", opponent=" LAC ", season=2026, week=1, game_id=" 2026_01_KC_LAC ",
        market_key="player_pass_yds", sportsbook=" draftkings ", line=250.5,
        selection=" OVER ", selected_price=-110, recorded_at=RECORDED_TIME,
        odds_retrieved_at=ODDS_TIME, research_notes="  matchup note  ",
    )
    values.update(changes)
    return PendingDecision(**values)


def rb_decision(**changes):
    values = dict(
        decision_id="decision-rb", position="RB", player_id="rb-1", player_name="Runner",
        team="BUF", opponent="MIA", season=2026, week=1, game_id="2026_01_BUF_MIA",
        market_key="player_rush_yds", sportsbook="fanduel", line=Decimal("67.5"),
        selection="under", selected_price=100, recorded_at=RECORDED_TIME + timedelta(minutes=1),
        odds_retrieved_at=ODDS_TIME, research_notes=None,
    )
    values.update(changes)
    return PendingDecision(**values)


def test_schema_initialization_is_idempotent():
    connection = sqlite3.connect(":memory:")
    initialize_decision_store(connection)
    record_decision(connection, qb_decision())
    initialize_decision_store(connection)
    assert get_decision(connection, "decision-qb") is not None


def test_qb_over_round_trips_with_exact_snapshot(connection):
    stored = record_decision(connection, qb_decision())
    assert stored == get_decision(connection, "decision-qb")
    assert stored.position == "QB"
    assert stored.sportsbook == "draftkings"
    assert stored.research_notes == "matchup note"
    assert stored.status == "pending" and stored.actual_result is None and stored.settled_at is None


def test_rb_under_round_trips(connection):
    stored = record_decision(connection, rb_decision())
    assert stored.position == "RB"
    assert stored.market_key == "player_rush_yds"
    assert stored.selection == "under"


def test_line_price_and_decimal_round_trip_canonically(connection):
    stored = record_decision(connection, qb_decision(line=Decimal("250.500"), selected_price=-115))
    assert stored.line == Decimal("250.5")
    assert get_decision(connection, "decision-qb").line == Decimal("250.5")
    assert stored.selected_price == -115


def test_blank_notes_become_null(connection):
    record_decision(connection, qb_decision(research_notes="  "))
    assert get_decision(connection, "decision-qb").research_notes is None


def test_duplicate_identity_is_rejected_but_changed_line_price_and_side_are_allowed(connection):
    record_decision(connection, qb_decision())
    with pytest.raises(DuplicateDecisionError, match="duplicate decision"):
        record_decision(connection, qb_decision(decision_id="other-id"))
    record_decision(connection, qb_decision(decision_id="line", line=251.5))
    record_decision(connection, qb_decision(decision_id="price", selected_price=-105))
    record_decision(connection, qb_decision(decision_id="under", selection="under"))
    assert [item.decision_id for item in list_decisions(connection)] == ["decision-qb", "line", "price", "under"]


def test_reused_decision_id_is_rejected(connection):
    record_decision(connection, qb_decision())
    with pytest.raises(DecisionIdConflictError, match="decision_id"):
        record_decision(connection, rb_decision(decision_id="decision-qb"))


@pytest.mark.parametrize("position,market_key", [("QB", "player_rush_yds"), ("RB", "player_pass_yds"), ("WR", "player_pass_yds")])
def test_invalid_position_market_combinations_are_rejected(connection, position, market_key):
    with pytest.raises(DecisionValidationError, match="position and market_key"):
        record_decision(connection, qb_decision(position=position, market_key=market_key))


@pytest.mark.parametrize("change, message", [
    ({"selection": "yes"}, "selection"), ({"status": "win"}, "status"),
    ({"actual_result": 250}, "actual_result"), ({"settled_at": RECORDED_TIME}, "settled_at"),
])
def test_recording_only_accepts_pending_null_result_fields(connection, change, message):
    with pytest.raises(DecisionValidationError, match=message):
        record_decision(connection, qb_decision(**change))


@pytest.mark.parametrize("price", [0, 99, -99])
def test_invalid_american_odds_are_rejected(connection, price):
    with pytest.raises(DecisionValidationError, match="selected_price"):
        record_decision(connection, qb_decision(selected_price=price))


@pytest.mark.parametrize("price", [100, -100])
def test_boundary_american_odds_are_accepted(connection, price):
    assert record_decision(connection, qb_decision(selected_price=price)).selected_price == price


@pytest.mark.parametrize("field, value", [("season", True), ("week", False), ("selected_price", True), ("line", True)])
def test_boolean_numeric_inputs_are_rejected(connection, field, value):
    with pytest.raises(DecisionValidationError):
        record_decision(connection, qb_decision(**{field: value}))


@pytest.mark.parametrize("field, value", [("season", 0), ("season", -1), ("week", 0), ("week", -1)])
def test_nonpositive_season_and_week_are_rejected(connection, field, value):
    with pytest.raises(DecisionValidationError, match=field):
        record_decision(connection, qb_decision(**{field: value}))


@pytest.mark.parametrize("field", [
    "decision_id", "position", "player_id", "player_name", "team", "opponent",
    "game_id", "market_key", "sportsbook", "selection", "status",
])
def test_blank_required_text_is_rejected(connection, field):
    with pytest.raises(DecisionValidationError, match=field):
        record_decision(connection, qb_decision(**{field: "  "}))


@pytest.mark.parametrize("line", [float("nan"), float("inf"), float("-inf"), Decimal("NaN")])
def test_nonfinite_lines_are_rejected(connection, line):
    with pytest.raises(DecisionValidationError, match="line"):
        record_decision(connection, qb_decision(line=line))


def test_naive_timestamps_and_backward_recorded_time_are_rejected(connection):
    with pytest.raises(DecisionValidationError, match="recorded_at"):
        record_decision(connection, qb_decision(recorded_at=datetime(2026, 9, 10, 17, 5)))
    with pytest.raises(DecisionValidationError, match="odds_retrieved_at"):
        record_decision(connection, qb_decision(odds_retrieved_at=datetime(2026, 9, 10, 17)))
    with pytest.raises(DecisionValidationError, match="must not precede"):
        record_decision(connection, qb_decision(recorded_at=ODDS_TIME - timedelta(seconds=1)))


def test_timestamps_normalize_to_utc_and_factories_are_injectable(connection):
    chicago_time = datetime(2026, 9, 10, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
    stored = record_decision(
        connection, qb_decision(decision_id=None, recorded_at=None, odds_retrieved_at=chicago_time),
        id_factory=lambda: "factory-id", clock=lambda: RECORDED_TIME,
    )
    assert stored.decision_id == "factory-id"
    assert stored.odds_retrieved_at == ODDS_TIME
    assert stored.recorded_at.tzinfo == timezone.utc


def test_missing_listing_order_rejected_insert_and_connection_lifetime(connection):
    assert get_decision(connection, "missing") is None
    record_decision(connection, qb_decision(decision_id="b", recorded_at=RECORDED_TIME + timedelta(minutes=1)))
    record_decision(connection, rb_decision(decision_id="a", recorded_at=RECORDED_TIME + timedelta(minutes=1)))
    assert [item.decision_id for item in list_decisions(connection)] == ["a", "b"]
    before = len(list_decisions(connection))
    with pytest.raises(DecisionValidationError):
        record_decision(connection, qb_decision(decision_id="bad", line=float("nan")))
    assert len(list_decisions(connection)) == before
    connection.execute("SELECT 1").fetchone()


def test_module_exposes_no_general_update_or_upsert_api():
    assert not hasattr(decisions, "update_decision")
    assert not hasattr(decisions, "upsert_decision")


def test_schema_allows_only_consistent_future_settlement_states(connection):
    record_decision(connection, qb_decision())
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE decisions SET status = 'win' WHERE decision_id = 'decision-qb'")
    connection.execute(
        """
        UPDATE decisions
        SET status = ?, actual_result = ?, settled_at = ?
        WHERE decision_id = ?
        """,
        ("win", "251", "2026-09-11T00:00:00Z", "decision-qb"),
    )
    stored = get_decision(connection, "decision-qb")
    assert stored.status == "win"
    assert stored.actual_result == "251"
    assert stored.settled_at == datetime(2026, 9, 11, tzinfo=timezone.utc)
