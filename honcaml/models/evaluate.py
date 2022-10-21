import pandas as pd
import sklearn.metrics as sk_metrics
from typing import Dict

from honcaml.models import base as base_model, general
from honcaml.tools import custom_typing as ct
from honcaml.tools.startup import logger


def cross_validate_model(
        model: base_model.BaseModel, x: ct.Array, y: ct.Array,
        cv_split: ct.SklearnCrossValidation, train_settings: Dict = None,
        test_settings: Dict = None) -> Dict:
    """
    This function trains a model with a cross-validation strategy. for each
    split, it trains a model and computes metrics. Finally, it computes the
    mean metrics.

    Args:
        model (base_model.BaseModel): a model to train with cross-validation.
        x (ct.Array): dataset features.
        y (ct.Array): dataset target.
        cv_split (ct.SklearnCrossValidation): cross-validation instance to
            apply during training step.
        train_settings (Dict): additional parameters for train step.
        test_settings (Dict): additional parameters for test step.

    Returns:
        (Dict): a dict with mean metrics.
    """
    if train_settings is None:
        train_settings = {}
    if test_settings is None:
        test_settings = {}

    results = []
    for split, x_train, x_test, y_train, y_test in cv_split.split(x, y):
        logger.info(f'Running split {split}/{cv_split.n_splits} ...')
        model.fit(x_train, y_train, **train_settings)
        results.append(model.evaluate(x_test, y_test, **test_settings))
    logger.info('Done.')
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


def compute_classifier_metrics(
        y_true: pd.Series, y_predicted: pd.Series) -> Dict[str, ct.Number]:
    """
    This function computes a classifier report dictionary containing most
    used classification metrics such as accuracy, precision, recall, etc.

    Args:
        y_true (pd.Series): series of ground truth classes
        y_predicted (pd.Series): series of predicted classes

    Returns:
        Dict containing computed metrics
    """
    tn, fp, fn, tp = sk_metrics.confusion_matrix(y_true, y_predicted).ravel()
    metrics = {
        'accuracy': sk_metrics.accuracy_score(y_true, y_predicted),
        'precision': sk_metrics.precision_score(y_true, y_predicted),
        'sensitivity': sk_metrics.recall_score(y_true, y_predicted),
        'specificity': tn / (tn + fp),
        'f1': sk_metrics.f1_score(y_true, y_predicted),
        'roc_auc': sk_metrics.roc_auc_score(y_true, y_predicted),
    }

    return metrics
