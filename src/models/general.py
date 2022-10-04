from typing import Dict, List
from src.tools import custom_typing as ct
from src.models import base as base_model, sklearn_model, general
from src.data import base as base_dataset
from src.exceptions import model as model_exceptions
import sklearn.metrics as sk_metrics
import pandas as pd


def initialize_model(model_type: str, estimator_type: str) \
        -> base_model.BaseModel:
    """
    Initialize the specific type of model.

    Args:
        model_type: the kind of model to initialize.
        estimator_type: the kind of estimator to be used. Valid
            values are `regressor` and `classifier`.

    Returns:
        model: the requested model instance.
    """
    if model_type == base_model.ModelType.sklearn:
        model = sklearn_model.SklearnModel(estimator_type)
    else:
        raise model_exceptions.ModelDoesNotExists(model_type)
    return model


def cross_validate_model(model: base_model.BaseModel,
                         x: ct.Array, y: ct.Array,
                         cv_split: ct.SklearnCrossValidation,
                         train_settings: Dict, test_settings: Dict) -> Dict:
    results = []
    for split, x_train, x_test, y_train, y_test in cv_split.split(x, y):
        model.fit(x_train, y_train, **train_settings)
        results.append(model.evaluate(x_test, y_test, **test_settings))
    # Group cv metrics
    cv_results = general.aggregate_cv_results(results)
    return cv_results


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
