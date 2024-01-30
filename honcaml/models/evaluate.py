import pandas as pd
import numpy as np
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
    Metrics values can be either strings or dictionaries.
    The way to compute them is depending on their name (string or key,
    respectively):
    - If there is a function in this same script named
    'compute_{metric}_metric', replacing {metric} for the given name, this
    function is used.
    - If not, it is assumed that metric can be directly drawn from
    sklearn.metrics module.
    In any case, if metric value is a single string, it means to pass no
    parameters to the function, whereas if it is a dictionary, its values are
    the set of parameters to pass.

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
        metric_params = {}
        # In case metric is a dict, get function name and parameters
        if isinstance(metric, dict):
            metric_params = list(metric.values())[0]
            metric = list(metric.keys())[0]
        try:
            metric_func_name = '_'.join(['compute', metric, 'metric'])
            metric_function = globals()[metric_func_name]
        except KeyError:
            try:
                metric_function = getattr(sk_metrics, metric)
            except AttributeError:
                raise model_exceptions.MetricDoesNotExist(metric)
        metrics_results[metric] = metric_function(
            y_true, y_predicted, **metric_params)
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


def compute_roc_auc_score_metric(
        y_true: pd.Series, y_predicted: pd.Series, **kwargs) -> ct.Number:
    """
    Calculate roc_auc metric from true values and predictions.

    Args:
        y_true: Ground truth outputs.
        y_predicted: Predicted outputs.

    Returns:
        Resulting metric.
    """

    classes = np.unique(y_true)
    n_classes = len(classes)

    df = pd.DataFrame(y_true, columns=['Valor'])
    one_hot_encoded_true = pd.get_dummies(df['Valor'])
    y_test_binarized = np.array(one_hot_encoded_true.values.tolist())

    df2 = pd.DataFrame(y_predicted, columns=['Valor'])
    one_hot_encoded_pred = pd.get_dummies(df2['Valor'])
    cols_to_add = list(set(one_hot_encoded_true.columns).difference(
        set(one_hot_encoded_pred)))
    for i in cols_to_add:
        one_hot_encoded_pred[i] = 0
    y_pred_binarized = np.array(one_hot_encoded_pred.values.tolist())

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sk_metrics.roc_curve(
            y_test_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[i] = sk_metrics.auc(fpr[i], tpr[i])
        roc_auc_score = sum(roc_auc.values()) / n_classes

    return roc_auc_score
