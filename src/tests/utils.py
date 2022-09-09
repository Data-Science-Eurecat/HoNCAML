import pandas as pd
from typing import Dict

from src.models import base as base_model
from src.models import sklearn_model
from src.exceptions import model as model_exceptions


def mock_up_read_dataframe() -> pd.DataFrame:
    """
    This method generates a dataframe for testing purposes.

    Returns:
        (pd.DataFrame): fake dataframe
    """
    data = {
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'target1': [10, 20, 30],
        'target2': [40, 50, 60]
    }
    return pd.DataFrame(data)


def mock_up_read_model(model_type: str, estimator_type: str,
                       model_config: Dict) -> base_model.BaseModel:
    """
    This method generates a model based on the type set for testing purposes.

    Args:
        model_type (str): the kind of model to fake.
        model_type (str): the kind of estimator to fake.
        model_config (Dict): the estimator config to fake.

    Returns:
        model (base.BaseModel): the fake model.
    """
    if model_type == base_model.ModelType.sklearn:
        model = sklearn_model.SklearnModel(estimator_type)
    else:
        raise model_exceptions.ModelDoesNotExists(model_type)
    model.build_model(model_config, {})
    return model