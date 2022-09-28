<<<<<<<< HEAD:honcaml/models/evaluate.py
========
from typing import Dict, List
from honcaml.tools import custom_typing as ct
import sklearn.metrics as sk_metrics
>>>>>>>> 9bcf8e0 (First version of documentation working):honcaml/models/general.py
import pandas as pd
import sklearn.metrics as sk_metrics
from typing import Dict, List

from src.tools import custom_typing as ct


def aggregate_cv_results(cv_results: List[Dict]) -> Dict:
    """
    Compute the cross validation results given a list of results containing
    the evaluated metrics for each partition of the data.

    Args:
        cv_results (List[Dict]): List of results containing each evaluated
            metric.

    Returns:
        mean_results (dict): The averaged metrics from the data partitions.
    """
    df_results = pd.DataFrame(cv_results)
    mean_results = df_results.mean(axis=0).to_dict()
    return mean_results


def compute_regression_metrics(
        y_true: pd.Series, y_predicted: pd.Series) -> Dict[str, ct.Number]:
    """
    This function computes regression metrics including 'max_error', 'MAPE',
    'MAE' and other commonly used regression metrics.

    Args:
        y_true (pd.Series): series of ground truth outputs
        y_predicted (pd.Series): series of predicted outputs

    Returns:
        metrics (Dict): the resulting metrics.
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
    This function computes classification metrics including [...] and other
    commonly used classification metrics.

    Args:
        y_true (pd.Series): series of ground truth outputs
        y_predicted (pd.Series): series of predicted outputs

    Returns:
        metrics (Dict): the resulting metrics.
    """
    raise NotImplementedError('Not implemented yet')
