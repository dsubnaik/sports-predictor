import pytest

from sklearn.metrics import r2_score

from training.evaluate import evaluate_predictions


def test_evaluate_predictions():
    y_true = [5, 7, 3, 8]
    y_pred = [4, 8, 4, 7]

    metrics = evaluate_predictions(y_true, y_pred)

    assert isinstance(metrics, dict)

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["r2"] == pytest.approx(
        r2_score(y_true, y_pred)
    )