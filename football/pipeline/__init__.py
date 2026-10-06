"""Reusable football research pipelines."""

from football.pipeline.build_weekly_qb_research import (
    WeeklyQBResearchResult,
    build_weekly_qb_research,
)
from football.pipeline.build_weekly_rb_research import (
    WeeklyRBResearchResult,
    build_weekly_rb_research,
)
from football.pipeline.build_weekly_player_prop_odds import (
    WEEKLY_PLAYER_PROP_ODDS_COLUMNS,
    WeeklyPlayerPropOddsResult,
    build_weekly_player_prop_odds,
)
from football.pipeline.build_selected_game_player_prop_odds import (
    ImmutableOddsTable,
    SelectedGamePlayerPropOddsResult,
    build_selected_game_player_prop_odds,
)
from football.pipeline.qb_passing_yards_weekly_inference import (
    QBPassingYardsWeeklyFeatureRow,
    QBPassingYardsWeeklyFeatureRows,
    QBPassingYardsWeeklyProjection,
    QBPassingYardsWeeklyProjectionReport,
    QBPassingYardsWeeklySkip,
    build_qb_passing_yards_weekly_feature_rows,
    generate_qb_passing_yards_weekly_projections,
)
from football.pipeline.qb_passing_yards_weekly_projection_snapshots import (
    QBPassingYardsWeeklyProjectionSnapshot,
    QBPassingYardsWeeklyProjectionSnapshotArtifact,
    QBPassingYardsWeeklyProjectionSnapshotMetadata,
    QBPassingYardsWeeklyProjectionSnapshotProjection,
    QBPassingYardsWeeklyProjectionSnapshotSkip,
    load_qb_passing_yards_weekly_projection_snapshot,
    save_qb_passing_yards_weekly_projection_snapshot,
)

__all__ = [
    "WeeklyQBResearchResult",
    "build_weekly_qb_research",
    "WeeklyRBResearchResult",
    "build_weekly_rb_research",
    "WEEKLY_PLAYER_PROP_ODDS_COLUMNS",
    "WeeklyPlayerPropOddsResult",
    "build_weekly_player_prop_odds",
    "ImmutableOddsTable",
    "SelectedGamePlayerPropOddsResult",
    "build_selected_game_player_prop_odds",
    "QBPassingYardsWeeklySkip",
    "QBPassingYardsWeeklyFeatureRow",
    "QBPassingYardsWeeklyFeatureRows",
    "QBPassingYardsWeeklyProjection",
    "QBPassingYardsWeeklyProjectionReport",
    "build_qb_passing_yards_weekly_feature_rows",
    "generate_qb_passing_yards_weekly_projections",
    "QBPassingYardsWeeklyProjectionSnapshotArtifact",
    "QBPassingYardsWeeklyProjectionSnapshotSkip",
    "QBPassingYardsWeeklyProjectionSnapshotProjection",
    "QBPassingYardsWeeklyProjectionSnapshotMetadata",
    "QBPassingYardsWeeklyProjectionSnapshot",
    "save_qb_passing_yards_weekly_projection_snapshot",
    "load_qb_passing_yards_weekly_projection_snapshot",
]
