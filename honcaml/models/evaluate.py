from typing import Dict, List
from honcaml.tools import custom_typing as ct
import sklearn.metrics as sk_metrics
import pandas as pd


def aggregate_cv_results(cv_results: List[Dict]) -> Dict:
    """
    Computes cross validation results given a list of results containing
    the evaluated metrics for each partition of the data.

    Args:
        cv_results: Results containing each evaluated metric.

    Returns:
        Averaged metrics from data partitions.
    """
    df_results = pd.DataFrame(cv_results)
    mean_results = df_results.mean(axis=0).to_dict()
    return mean_results


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
