from typing import Dict
from honcaml.tools import custom_typing as ct
from honcaml.models import base as base_model, general
import sklearn.metrics as sk_metrics
import pandas as pd


def cross_validate_model(
        model: base_model.BaseModel, x: ct.Array, y: ct.Array,
        cv_split: ct.SklearnCrossValidation, train_settings: Dict = None,
        test_settings: Dict = None) -> Dict:
    if train_settings is None:
        train_settings = {}
    if test_settings is None:
        test_settings = {}
    results = []
    for split, x_train, x_test, y_train, y_test in cv_split.split(x, y):
        model.fit(x_train, y_train, **train_settings)
        results.append(model.evaluate(x_test, y_test, **test_settings))
    # Group cv metrics
    cv_results = general.aggregate_cv_results(results)
    return cv_results


def compute_regression_metrics(
        y_true: pd.Series, y_predicted: pd.Series) -> Dict[str, ct.Number]:
    """
    Computes regression metrics from true values and predictions; available
    options are specified within.

    Args:
        y_true: Ground truth outputs.
        y_predicted: Predicted outputs.

    Returns:
        Resulting metrics.
    """
    metrics = {
        # 'max_error': sk_metrics.max_error(y_true, y_predicted),
        'mean_squared_error': sk_metrics.mean_squared_error(
            y_true, y_predicted),
        'mean_absolute_percentage_error':
            sk_metrics.mean_absolute_percentage_error(y_true, y_predicted),
        'median_absolute_error': sk_metrics.median_absolute_error(
            y_true, y_predicted),
        'r2_score': sk_metrics.r2_score(y_true, y_predicted),
        'mean_absolute_error': sk_metrics.mean_absolute_error(
            y_true, y_predicted),
        'root_mean_square_error': sk_metrics.mean_squared_error(
            y_true, y_predicted, squared=False),
    }

    return metrics


def compute_classification_metrics(
        y_true: pd.Series, y_predicted: pd.Series) -> Dict[str, ct.Number]:
    """
    Computes classification metrics from true values and predictions; available
    options are specified within.

    Args:
        y_true: Series of ground truth outputs.
        y_predicted: Series of predicted outputs.

    Returns:
        Resulting metrics.
    """
    raise NotImplementedError('Not implemented yet')
