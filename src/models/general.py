import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
from typing import Dict, List, Union

from src.tools import utils

Number = Union[float, int, str]


class Metrics:
    accuracy = 'sklearn.metrics.accuracy_score'
    f1 = 'sklearn.metrics.f1_score'


def compute_metrics(y_true: np.array, y_pred: np.array, metrics: List) -> Dict:
    results = {}
    for metric in metrics:
        results['metric'] = utils.import_library(
            getattr(Metrics, metric), {'y_true': y_true, 'y_pred': y_pred})
    return results


def aggregate_cv_results(cv_results: List[Dict]) -> Dict:
    df_results = pd.DataFrame(cv_results)
    mean_results = df_results.mean(axis=0).to_dict()
    return mean_results


def compute_regression_metrics(
        y_true: pd.Series, y_predicted: pd.Series) -> Dict[str, Number]:
    """
    This function computes regression metrics including 'max_error', 'MAPE',
    'MAE' and other commonly used regression metrics.

    Args:
        y_true (pd.Series): series of ground truth outputs
        y_predicted (pd.Series): series of predicted outputs

    Returns:
        a dict containing computed metrics

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
        y_true: pd.Series, y_predicted: pd.Series) -> Dict[str, Number]:
    """
    This function computes classification metrics including [...] and other
    commonly used classification metrics.

    Args:
        y_true (pd.Series): series of ground truth outputs
        y_predicted (pd.Series): series of predicted outputs

    Returns:
        a dict containing computed metrics

    """
    raise NotImplementedError('Not implemented yet')
