"""Unit tests for append-only pending football decision recording."""

import sqlite3
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import football.decisions as decisions
from football.decisions import (
    DecisionIdConflictError,
    DecisionNotFoundError,
    DecisionValidationError,
    DuplicateDecisionError,
    PendingDecision,
    get_decision,
    initialize_decision_store,
    list_decisions,
    record_decision,
    settle_decision,
    SettlementConflictError,
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
    assert stored.actual_result == Decimal("251")
    assert stored.settled_at == datetime(2026, 9, 11, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "decision_factory, result, expected_status",
    [
        (qb_decision, Decimal("251"), "win"),
        (qb_decision, Decimal("250"), "loss"),
        (qb_decision, Decimal("250.5"), "push"),
        (rb_decision, Decimal("67"), "win"),
        (rb_decision, Decimal("68"), "loss"),
        (rb_decision, Decimal("67.5"), "push"),
    ],
)
def test_settlement_calculates_over_and_under_outcomes(
    connection, decision_factory, result, expected_status
):
    decision = decision_factory()
    record_decision(connection, decision)
    settled = settle_decision(
        connection,
        decision.decision_id,
        result,
        settled_at=RECORDED_TIME + timedelta(hours=1),
    )
    assert settled.status == expected_status
    assert settled.actual_result == result


def test_settlement_uses_exact_decimal_comparison_and_round_trips_result(connection):
    record_decision(connection, qb_decision(line=Decimal("0.3")))
    settled = settle_decision(
        connection,
        "decision-qb",
        Decimal("0.3000000000000000000000000000"),
        settled_at=RECORDED_TIME,
    )
    assert settled.status == "push"
    assert settled.actual_result == Decimal("0.3")
    assert get_decision(connection, "decision-qb").actual_result == Decimal("0.3")
    assert list_decisions(connection)[0].actual_result == Decimal("0.3")


def test_settlement_compares_negative_official_results_without_restriction(connection):
    record_decision(connection, qb_decision(line=Decimal("-1")))
    settled = settle_decision(
        connection, "decision-qb", Decimal("-0.5"), settled_at=RECORDED_TIME
    )
    assert settled.status == "win"


def test_settlement_timestamp_normalizes_to_utc(connection):
    record_decision(connection, qb_decision())
    central_time = datetime(2026, 9, 10, 13, 5, tzinfo=timezone(timedelta(hours=-5)))
    settled = settle_decision(connection, "decision-qb", 251, settled_at=central_time)
    assert settled.settled_at == datetime(2026, 9, 10, 18, 5, tzinfo=timezone.utc)
    assert settled.settled_at.tzinfo == timezone.utc


def test_settlement_clock_is_injectable(connection):
    record_decision(connection, qb_decision())
    settled = settle_decision(
        connection,
        "decision-qb",
        251,
        clock=lambda: RECORDED_TIME + timedelta(minutes=10),
    )
    assert settled.settled_at == RECORDED_TIME + timedelta(minutes=10)


def test_invalid_settlement_timestamps_are_rejected_without_changes(connection):
    before = record_decision(connection, qb_decision())
    with pytest.raises(DecisionValidationError, match="timezone-aware"):
        settle_decision(connection, "decision-qb", 251, settled_at=datetime(2026, 9, 10, 18))
    with pytest.raises(DecisionValidationError, match="must not precede"):
        settle_decision(connection, "decision-qb", 251, settled_at=RECORDED_TIME - timedelta(seconds=1))
    assert get_decision(connection, "decision-qb") == before


@pytest.mark.parametrize("result", [float("nan"), float("inf"), float("-inf"), Decimal("NaN"), True])
def test_invalid_actual_results_are_rejected_without_changes(connection, result):
    before = record_decision(connection, qb_decision())
    with pytest.raises(DecisionValidationError, match="actual_result"):
        settle_decision(connection, "decision-qb", result, settled_at=RECORDED_TIME)
    assert get_decision(connection, "decision-qb") == before


def test_settlement_rejects_empty_or_missing_decision_identifier(connection):
    with pytest.raises(DecisionValidationError, match="decision_id"):
        settle_decision(connection, "  ", 251, settled_at=RECORDED_TIME)
    with pytest.raises(DecisionNotFoundError, match="not found"):
        settle_decision(connection, "missing", 251, settled_at=RECORDED_TIME)


def test_settlement_changes_only_allowed_fields_and_preserves_snapshot(connection):
    before = record_decision(connection, qb_decision())
    settled = settle_decision(connection, "decision-qb", 251, settled_at=RECORDED_TIME)
    allowed = {"status", "actual_result", "settled_at"}
    for field in fields(before):
        if field.name not in allowed:
            assert getattr(settled, field.name) == getattr(before, field.name)
    assert settled.status == "win"
    assert settled.actual_result == Decimal("251")
    assert settled.settled_at == RECORDED_TIME


def test_exact_settlement_retry_is_idempotent_and_keeps_original_timestamp(connection):
    record_decision(connection, qb_decision())
    initial = settle_decision(
        connection, "decision-qb", Decimal("251.00"), settled_at=RECORDED_TIME
    )
    retried = settle_decision(
        connection,
        " decision-qb ",
        Decimal("251.0"),
        settled_at=RECORDED_TIME + timedelta(days=1),
    )
    assert retried == initial
    assert retried.settled_at == RECORDED_TIME


def test_conflicting_settlement_retry_and_inconsistent_settlement_are_rejected(connection):
    record_decision(connection, qb_decision())
    initial = settle_decision(connection, "decision-qb", 251, settled_at=RECORDED_TIME)
    with pytest.raises(SettlementConflictError, match="conflicts"):
        settle_decision(connection, "decision-qb", 252, settled_at=RECORDED_TIME)
    assert get_decision(connection, "decision-qb") == initial

    connection.execute(
        "UPDATE decisions SET actual_result = ? WHERE decision_id = ?",
        ("249", "decision-qb"),
    )
    with pytest.raises(SettlementConflictError, match="inconsistent"):
        settle_decision(connection, "decision-qb", 249, settled_at=RECORDED_TIME)
    assert get_decision(connection, "decision-qb").actual_result == Decimal("249")


def test_failed_database_settlement_does_not_leave_a_partial_update(connection):
    before = record_decision(connection, qb_decision())
    connection.execute(
        """
        CREATE TRIGGER reject_decision_settlement
        BEFORE UPDATE ON decisions
        BEGIN
            SELECT RAISE(ABORT, 'test settlement failure');
        END
        """
    )
    with pytest.raises(decisions.DecisionStoreError, match="database constraint"):
        settle_decision(connection, "decision-qb", 251, settled_at=RECORDED_TIME)
    assert get_decision(connection, "decision-qb") == before


def test_connection_remains_usable_after_settlement_success_and_failure(connection):
    record_decision(connection, qb_decision())
    settle_decision(connection, "decision-qb", 251, settled_at=RECORDED_TIME)
    connection.execute("SELECT 1").fetchone()
    with pytest.raises(SettlementConflictError):
        settle_decision(connection, "decision-qb", 252, settled_at=RECORDED_TIME)
    assert connection.execute("SELECT 1").fetchone() == (1,)
