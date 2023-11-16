import pandas as pd
from typing import Dict

from honcaml.data import normalization
from honcaml.models import base as base_model
from honcaml.models import sklearn_model, torch_model


def mock_up_yaml() -> Dict:
    yaml_content = \
        "{'key1': {'nest_key1': 1,'nest_key2': 2,},'key2': 'value',}"
    return yaml_content


def mock_up_read_pipeline() -> Dict:
    pipeline_content = {
        'global': {
            'problem_type': 'regression',
        },
        'steps': {
            'data': {},
            'model': {},
        },
    }
    return pipeline_content


def mock_up_read_dataframe() -> pd.DataFrame:
    """
    Generates a dataframe for testing purposes.

    Returns:
        Fake dataframe.
    """
    data = {
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'target': [10, 20, 30]
    }
    return pd.DataFrame(data)


def mock_up_read_classifier_dataframe() -> pd.DataFrame:
    """
    Generates a dataframe for testing purposes.

    Returns:
        Fake dataframe.
    """
    data = {
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'target': [0, 0, 1],
    }
    return pd.DataFrame(data)


def mock_up_read_model(model_type: str, problem_type: str,
                       model_config: Dict, norm_config: dict = None,
                       features: list = None, target: list = None) \
        -> base_model.BaseModel:
    """
    Generates a model based on the type set for testing purposes.

    Args:
        model_type: The kind of model to fake.
        problem_type: The kind of problem to fake.
        model_config: The estimator config to fake.
        norm_config: Normalizations configuration.
        features: List of columns used as features.
        target: List of columns considered targets.

    Returns:
        The fake model.
    """
    if norm_config is None:
        norm_config = {}
    norm = normalization.Normalization(norm_config)
    if model_type == base_model.ModelType.sklearn:
        model = sklearn_model.SklearnModel(problem_type)
    elif model_type == base_model.ModelType.torch:
        model = torch_model.TorchModel(problem_type)
    else:
        raise NotImplementedError('The model implementation does not exist ')

    model.build_model(model_config, norm, features, target)
    return model
