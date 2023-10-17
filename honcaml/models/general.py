import pandas as pd
from typing import Dict, List

from honcaml.exceptions import model as model_exceptions
from honcaml.models import (
    base as base_model,
    sklearn_model,
    torch_model
)


def initialize_model(model_type: str, problem_type: str) \
        -> base_model.BaseModel:
    """
    Initialize the specific type of model.

    Args:
        model_type: the kind of model to initialize.
        problem_type: the kind of problem to be addressed. Valid
            values are `regression` and `classification`.

    Returns:
        model: the requested model instance.
    """
    if model_type == base_model.ModelType.sklearn:
        model = sklearn_model.SklearnModel(problem_type)
    elif model_type == base_model.ModelType.torch:
        model = torch_model.TorchModel(problem_type)
    else:
        raise model_exceptions.ModelDoesNotExists(model_type)
    return model


def aggregate_cv_results(cv_results: List[Dict]) -> Dict:
    """
    Computes cross validation results given a list of results containing
    the evaluated metrics for each partition of the data.

    Args:
        cv_results: Results containing each evaluated metric.

    Returns:
        Averaged metrics from data partitions.
    """
    mean_results = pd.DataFrame(cv_results) \
        .mean(axis=0) \
        .to_dict()
    return mean_results
