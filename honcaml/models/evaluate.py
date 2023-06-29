import pandas as pd
import sklearn.metrics as sk_metrics
from typing import Dict, List
from honcaml.exceptions import model as model_exceptions

from honcaml.models import base as base_model, general
from honcaml.tools import custom_typing as ct


def cross_validate_model(
        model: base_model.BaseModel, x: ct.Array, y: ct.Array,
        cv_split: ct.SklearnCrossValidation, metrics: List,
        train_settings: Dict = None, test_settings: Dict = None,
        aggregate_metrics: bool = True) -> Dict:
    """
    This function trains a model with a cross-validation strategy. for each
    split, it trains a model and computes metrics. Finally, it computes the
    aggregated metrics.

    Args:
        model (base_model.BaseModel): a model to train with cross-validation.
        x (ct.Array): dataset features.
        y (ct.Array): dataset target.
        cv_split (ct.SklearnCrossValidation): cross-validation instance to
            apply during training step.
        metrics (List): Metrics to be computed.
        train_settings (Dict): additional parameters for train step.
        test_settings (Dict): additional parameters for test step.
        aggregate_metrics (bool): Whether to aggregate resulting metrics.

    Returns:
        (Dict): a dict with mean metrics.
    """
    if train_settings is None:
        train_settings = {}
    if test_settings is None:
        test_settings = {}

    cv_results = []
    for split, x_train, x_test, y_train, y_test in cv_split.split(x, y):
        model.fit(x_train, y_train, **train_settings)
        cv_results.append(model.evaluate(x_test, y_test, metrics,
                                         **test_settings))
    # Group cv metrics
    if aggregate_metrics:
        cv_results = general.aggregate_cv_results(cv_results)

    return cv_results


def compute_metrics(
        y_true: pd.Series, y_predicted: pd.Series,
        metrics: list) -> Dict[str, ct.Number]:
    """
    Computes specified metrics from true values and predictions.
    For each metric, there are two possible ways to be computed:
    - If there is a function in this same script named
    'compute_{metric}_metric', replacing {metric} for the given name, this
    function is used.
    - If not, it is assumed that metric can be directly drawn from
    sklearn.metrics module, and its raw function is used without parameters.

    Args:
        y_true: Ground truth outputs.
        y_predicted: Predicted outputs.
        metrics: Metrics to be computed.

    Returns:
        Resulting metrics.

    Raises:
        AttributeError in case function is not found in any of the
        aforementioned possibilities
    """
    metrics_results = {}
    for metric in metrics:
        try:
            metric_func_name = '_'.join(['compute', metric, 'metric'])
            metric_function = globals()[metric_func_name]
        except KeyError:
            try:
                metric_function = getattr(sk_metrics, metric)
            except AttributeError:
                raise model_exceptions.MetricDoesNotExist(metric)
        metrics_results[metric] = metric_function(y_true, y_predicted)
    return metrics_results


def compute_root_mean_squared_error_metric(
        y_true: pd.Series, y_predicted: pd.Series) -> ct.Number:
    """
    Computes root mean squared error metric from true values and
    predictions.

    Args:
        y_true: Ground truth outputs.
        y_predicted: Predicted outputs.

    Returns:
        Resulting metric.
    """
    root_mean_squared_error = sk_metrics.mean_squared_error(
        y_true, y_predicted, squared=False)
    return root_mean_squared_error


def compute_specificity_score_metric(
        y_true: pd.Series, y_predicted: pd.Series, **kwargs) -> ct.Number:
    """
    Computes specificity metric from true values and predictions.

    Args:
        y_true: Ground truth outputs.
        y_predicted: Predicted outputs.

    Returns:
        Resulting metric.
    """
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_predicted).ravel()
    specificity = tn / (tn + fp)
    return specificity
