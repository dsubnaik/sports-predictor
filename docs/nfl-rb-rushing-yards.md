# NFL RB Rushing-Yards Modeling: Population Policy Proposal

## Status and scope

This is a proposed data-population policy for a future RB rushing-yards dataset. It documents what the current repository can support; it does not create a dataset, fit a model, generate predictions, download data, or freeze the policy.

The central distinction is intentional:

- An **outcome-observed research row** is an RB player-game retained by the completed-game player-stat source and having a finite rushing-yards target.
- A **pregame-eligible candidate** is a scheduled RB identified before the game from an admissible dated source. It need not have an observed outcome row, and an observed outcome row need not have reconstructable pregame eligibility.

Those sets answer different questions and must not be silently substituted for one another.

## 1. Architecture findings

### Reusable QB infrastructure

`football.training.build_qb_passing_yards_dataset.build_qb_passing_yards_training_dataset` is the closest dataset-builder pattern. It keys rows by `[season, week, game_id, player_id]`, attaches schedule context, retains a finite target, builds features before target-week outcomes, and validates unique keys. Its `_history_cutoff` uses the complete preceding season for Week 1 and the current season with `week < target_week` thereafter. It processes features by target row and excludes every game in the target week, regardless of kickoff time.

`football.training.split_qb_passing_yards_dataset.split_qb_passing_yards_dataset` is reusable as the chronology contract: rows before `validation_start` are train, rows from that boundary through (but excluding) `test_start` are validation, and later rows are test. The completed QB workflow fixes the boundaries at `(2025, 1)` and `(2026, 1)`, respectively:

| Partition | Exact QB target boundary |
| --- | --- |
| Train | 2021 through 2024 (before `(2025, 1)`) |
| Validation | `(2025, 1)` through before `(2026, 1)` |
| Holdout | `(2026, 1)` and later |

The splitter itself is generic and parameterized; those years are the QB workflow's configured boundaries, not an RB policy already implemented. `football.training.audit_qb_passing_yards_dataset.audit_qb_passing_yards_dataset` validates schema, keys, chronology, history indicators, and partition overlap. It reads targets only from train and validation; test structure can be audited without reading test targets. QB candidate comparison and selection fit on train only and evaluate validation only. After selection is frozen, `football.training.qb_passing_yards_production_model.train_qb_passing_yards_production_model` refits that frozen specification on train plus validation, never `dataset_split.test`; the protected 2026 holdout remains excluded. This test-blind lifecycle should be copied for RB work.

### RB-specific building blocks

The RB backend already provides the following public contracts, but no RB training-dataset builder or RB chronological splitter yet:

- `football.data.build_running_back_dataset.build_running_back_dataset` creates normalized regular-season RB player-game rows (`OUTPUT_COLUMNS`) from nflverse weekly player statistics.
- `football.features.running_back_usage.annotate_running_back_usage` preserves every normalized RB row and adds retrospective team totals, shares, leaders, low-volume, and shared-backfield flags.
- `football.features.rb_form_metrics.build_rb_form_metrics` produces prior-game player form summaries. `football.features.defense_rb_game_logs.build_defense_rb_game_logs` aggregates completed opponent RB production, and `football.features.rb_defense_matchup_metrics.build_rb_defense_matchup_metrics` produces strictly prior defense summaries and a historical matchup rank.
- `football.data.normalize_depth_charts.normalize_nflverse_depth_charts` maps dated 2025+ source snapshots to `CANONICAL_DEPTH_CHART_COLUMNS`: `team`, `player_id`, `player_name`, `position`, `snapshot_timestamp`, `depth_position`, and `depth_rank`. `football.features.expected_running_backs.resolve_expected_running_backs` retains every exact-`RB` entry from the latest eligible snapshot for each requested team, or emits an unresolved diagnostic row. It does not select a lead back and does not use usage.

The weekly research pipeline (`football.pipeline.build_weekly_rb_research.build_weekly_rb_research`) deliberately preserves all expected backfield participants and makes no prediction. Its `as_of_date` is a slate-wide dated depth-chart cutoff, rather than a game-specific kickoff cutoff. The report/UI layer is not a population filter: `football.ui.rb_research_view.build_participant_options` hides unresolved identities from the selector, while backend report rows retain unresolved rows and all resolved expected RBs. UI visibility must never define training membership or pregame eligibility.

### Roster, participation, and dated-depth evidence

The local RB loader surface (`football.data.fetch_nflverse`) exposes weekly player stats, schedules, and depth charts; it exposes no RB roster or participation loader/schema. Player statistics have keys including `season`, `week`, `game_id`, and `player_id`, but are postgame outcomes. The QB feasibility inventory in `football.training.qb_likely_primary_population_analysis.audit_qb_primary_qb_feasibility` records a weekly-roster schema `(season, week, team, player_id)` and legacy weekly depth-chart schema `(season, week, team, player_id)`, but says neither has a snapshot time that proves pre-kickoff availability. The RB depth-chart normalizer rejects legacy weekly schemas and accepts only the dated 2025+ schema. Therefore this repository does not establish historical pregame RB participant reconstruction across the QB training years; dated RB snapshots are suitable only where separately supplied and demonstrably prior to game time.

## 2. Historical target coverage

`build_running_back_dataset` retains source rows only when `position == "RB"` and `season_type == "REG"`; it retains one normalized row per observed source player-game and rejects conflicting duplicate keys. It does not construct a row for a player merely because that player was rostered, listed on a depth chart, scheduled, active, or participated in another way.

Synthetic fixtures establish that an observed RB row can have zero carries and zero rushing yards (`football.tests.test_build_running_back_dataset`), and that the usage layer retains both a zero-carry receiving RB and a zero-carry, zero-target, zero-opportunity RB (`football.tests.test_running_back_usage`). Thus an explicit normalized row with zero carries is an observed zero for that source row; it is valid for an outcome-observed population when its rushing yards are finite. A missing player-game is instead an absent record with an unknown outcome. It must not be imputed as zero rushing yards, zero carries, or nonparticipation.

The code and fixtures do not establish source completeness for every active RB, inactive RB, special-teams-only player, or player with no recorded stat line. They also do not establish historical pregame participant coverage. Claims and metrics must therefore be limited to represented, outcome-observed RB player-games unless a separately validated dated participant source is joined.

## 3. Proposed population policy

### Initial recommendation

For the first dataset, retain every normalized regular-season RB player-game that has schedule context and a finite observed `rushing_yards` target. Do not apply a target-game usage threshold or select a target-game team leader. This is an outcome-observed research population, not proof that each row was a pregame candidate.

For future deployment, define eligibility separately: scheduled team-games with a resolved RB from a dated depth-chart snapshot known before the relevant game's cutoff, or an explicitly dated/manual pregame override. Preserve all such RBs. Eligibility must not use target-game carries, targets, opportunities, shares, leader flags, low-volume flags, rushing yards, or any participation or box-score field.

Zero and low-volume observed games remain training/evaluation rows. Committee backs remain separate player-game rows; no primary-back label is required. Rookies, first appearances, injury returns, promotions, and players without eligible history remain rows with explicit cold-start/missing-history metadata, not exclusions. Injured or inactive players cannot be reliably removed from historical eligibility with current repository evidence. A team change preserves the player's player-ID history for player form, but team-relative shares must be calculated only from the player's prior rows with the team represented in each historical game; never reinterpret an old-team share as a new-team share.

This proposal does not freeze either population. The key unresolved gap is a game-time-qualified historical participant source. Without it, retrospective model evaluation can describe outcome-observed rows and pregame-defined slices only where dated snapshots exist; it cannot claim performance for all players who were truly eligible before every historical kickoff.

## 4. Dataset foundation specification

### Proposed row and target

Use `(season, week, game_id, player_id)` as the unique row key, with schedule metadata `player_name`, `team`, `opponent`, and `home_away`. The target is `target_rushing_yards`, copied only from the finite normalized `rushing_yards` outcome. Keep the target, identity, and descriptive context outside canonical model inputs.

Week 1 must use eligible completed regular-season history from the immediately prior season; Week 2+ must use completed target-season games with `week < target_week`. Compute a whole target week before admitting any of that week's outcomes. Do not include target-week or future targets, carries, targets, shares, defense results, or participation data. Apply the same cutoff to defense logs and ranks.

### Candidate input families

The eventual builder should publish a small, explicit canonical feature list selected only after validation. Candidate families and their present schemas are:

| Family | Existing source/schema | Proposed prior-only fields |
| --- | --- | --- |
| RB production and workload | `build_running_back_dataset.OUTPUT_COLUMNS`; `RUNNING_BACK_USAGE_COLUMNS` | season/last-three rushing-yards and attempts averages; history counts; season/last-three yards per carry; optionally prior opportunities, targets, and team-relative carry/opportunity-share averages |
| Defense versus RBs | `DEFENSE_RB_GAME_LOG_COLUMNS`; `rb_defense_matchup_metrics.OUTPUT_COLUMNS` | season/last-three RB rushing-yards and attempts allowed, defense history counts, and optionally matchup rank |
| Availability context | `CANONICAL_DEPTH_CHART_COLUMNS`; expected-RB `OUTPUT_COLUMNS` | only a separately validated pregame depth snapshot's rank/slot/source or missingness, if coverage is sufficient; not a target-game usage proxy |
| Schedule context | normalized schedule context | home/away only if selected after validation; no outcome-derived game-script fields |

Target-game `carry_share`, `opportunity_share`, leader flags, `low_volume_rb`, and `shared_backfield` are diagnostic-only retrospective fields. Their historical, strictly prior summaries may be candidate inputs, but must never filter row membership. Sportsbook lines, odds, prices, and decision records are outside features and eligibility.

### Diagnostics, missingness, and team changes

Keep diagnostic/slice columns separate from canonical inputs: target key, target, names/teams, target-week participant-resolution status/source, depth metadata, player and defense history counts, cold-start flags, precomputed prior-only workload/committee categories, and team-change flags. Target-week labels must be defined from pregame sources only; retrospective leader/low-volume labels belong only to observed-outcome diagnostics.

Represent no usable eligible prior player history with null historical averages, zero history counts, and `rb_missing_history=True`; do not turn it into a zero-performance history. Treat a player whose latest prior team differs from target `team` as `player_team_changed=True`. Retain player-level form by stable `player_id`, expose the change flag, and compute team-relative shares from their original historical team-games. A separately chosen training-only imputer may handle canonical numeric nulls later; it must not alter the raw meaning of cold start.

## 5. Evaluation and next task

Use the same train/validation/test chronology and test-blind lifecycle as QB once RB boundaries are approved: fit/select candidates on train with validation-only evaluation, then refit a frozen specification on train plus validation while continuing to exclude the holdout. Report overall metrics only for the outcome-observed target population, then report pregame-defined slices where valid dated eligibility is available:

- leading versus secondary candidate by pregame depth rank/order, with `unknown` retained as its own slice;
- pregame expected one-RB versus multi-RB backfields;
- prior-only workload/history bands (including zero-history and limited-history flags), not target-game workload;
- prior-only committee/share bands; and
- player team-change versus unchanged-team rows.

These evaluations could support claims about error on represented, outcome-observed RB rows and, for the dated-snapshot subset, on the stated pregame candidate definition. They cannot establish performance for all true actives, all rostered players, or unobserved player-games.

The smallest next implementation task after policy review is a test-driven, local-only `build_rb_rushing_yards_training_dataset` module. It should accept injected normalized RB rows and schedule rows, emit the proposed raw row contract with finite observed targets and prior-only features, and add focused synthetic tests for Week 1, same-week exclusion, zero rows, cold starts, and a team change. Do not add splitting, models, depth eligibility joins, or live loading in that task.
