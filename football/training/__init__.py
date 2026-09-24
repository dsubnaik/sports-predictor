"""Football model-training dataset builders and temporal splits."""

from football.training.split_qb_passing_yards_dataset import (
    QBPassingYardsDatasetSplit,
    split_qb_passing_yards_dataset,
)
from football.training.qb_passing_yards_baseline import (
    QBPassingYardsBaseline,
    RegressionMetrics,
    evaluate_qb_passing_yards_predictions,
    fit_qb_passing_yards_baseline,
    predict_qb_passing_yards_baseline,
)
from football.training.audit_qb_passing_yards_dataset import (
    QBPassingYardsDatasetAudit,
    PartitionSummary,
    SampleSizeSummary,
    TrainingFeatureSummary,
    TrainingTargetSummary,
    ValidationBaselineAudit,
    audit_qb_passing_yards_dataset,
)
from football.training.qb_passing_yards_linear_regression import (
    HistoricalAverageComparison,
    LinearRegressionCoefficient,
    NumericImputationSummary,
    QBPassingYardsLinearRegressionEvaluation,
    QBPassingYardsLinearRegressionModel,
    QBPassingYardsLinearRegressionValidation,
    evaluate_qb_passing_yards_linear_regression,
    fit_qb_passing_yards_linear_regression,
    predict_qb_passing_yards_linear_regression,
    train_and_validate_qb_passing_yards_linear_regression,
)

__all__ = [
    "QBPassingYardsDatasetSplit",
    "split_qb_passing_yards_dataset",
    "QBPassingYardsBaseline",
    "RegressionMetrics",
    "evaluate_qb_passing_yards_predictions",
    "fit_qb_passing_yards_baseline",
    "predict_qb_passing_yards_baseline",
    "QBPassingYardsDatasetAudit",
    "PartitionSummary",
    "SampleSizeSummary",
    "TrainingFeatureSummary",
    "TrainingTargetSummary",
    "ValidationBaselineAudit",
    "audit_qb_passing_yards_dataset",
    "HistoricalAverageComparison",
    "LinearRegressionCoefficient",
    "NumericImputationSummary",
    "QBPassingYardsLinearRegressionEvaluation",
    "QBPassingYardsLinearRegressionModel",
    "QBPassingYardsLinearRegressionValidation",
    "evaluate_qb_passing_yards_linear_regression",
    "fit_qb_passing_yards_linear_regression",
    "predict_qb_passing_yards_linear_regression",
    "train_and_validate_qb_passing_yards_linear_regression",
]
