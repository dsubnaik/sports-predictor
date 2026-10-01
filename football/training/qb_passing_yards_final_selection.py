"""Freeze the validation-selected QB passing-yards production candidate.

Selection fits only train and evaluates only validation; the held-out property
is deliberately never read.
"""
from dataclasses import dataclass
from football.training.build_qb_passing_yards_dataset import FEATURE_COLUMNS
from football.training.qb_passing_yards_gradient_boosting import (
    GradientBoostingParameters, QBPassingYardsGradientBoostingModel,
    QBPassingYardsGradientBoostingEvaluation,
    fit_qb_passing_yards_gradient_boosting,
    evaluate_qb_passing_yards_gradient_boosting,
)
from football.training.qb_passing_yards_baseline import RegressionMetrics
from football.training.split_qb_passing_yards_dataset import QBPassingYardsDatasetSplit

@dataclass(frozen=True)
class QBPassingYardsFinalSelectionReport:
    selected_model_name: str
    parameters: GradientBoostingParameters
    feature_columns: tuple[str, ...]
    numeric_preprocessing: str
    binary_preprocessing: str
    scaling: str
    validation_metrics: RegressionMetrics
    validation_evaluation: QBPassingYardsGradientBoostingEvaluation
    training_period: str
    validation_period: str
    holdout_status: str
    excluded_experiments: tuple[str, ...]
    limitations: tuple[str, ...]

@dataclass(frozen=True)
class QBPassingYardsFinalSelection:
    model: QBPassingYardsGradientBoostingModel
    report: QBPassingYardsFinalSelectionReport

def select_qb_passing_yards_final_model(dataset_split: QBPassingYardsDatasetSplit) -> QBPassingYardsFinalSelection:
    """Fit the frozen candidate on train and validate it; never access test."""
    model = fit_qb_passing_yards_gradient_boosting(dataset_split.train)
    evaluation = evaluate_qb_passing_yards_gradient_boosting(model, dataset_split.validation)
    return QBPassingYardsFinalSelection(model, QBPassingYardsFinalSelectionReport(
        "sklearn GradientBoostingRegressor", model.parameters, tuple(FEATURE_COLUMNS),
        "training-only median imputation", "validated boolean pass-through", "no scaling",
        evaluation.overall_metrics, evaluation, "2021–2024 train partition", "2025 validation partition",
        "2026 remains unopened and unscored",
        ("opponent-adjusted form", "fixed rank tiers", "depth-chart status", "sportsbook lines", "unverified style features"),
        ("MAE is an average validation error, not an individual prediction guarantee", "cold starts and non-primary appearances remain less reliable", "likely-primary selection is separate from prediction", "sportsbook profitability and style effects are untested"),
    ))
