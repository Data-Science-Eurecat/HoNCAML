import joblib
import os
import pandas as pd
import yaml
from typing import Dict

from honcaml.exceptions import data as data_exception
from honcaml.tools import custom_typing as ct
from honcaml.tools import utils
from honcaml.tools.startup import logger


def read_yaml(file_path: str) -> Dict:
    """
    Given a yaml file path, this function reads content and returns it.

    Args:
        file_path: Yaml's file path.

    Returns:
        Parsed information from file.
    """
    with open(file_path, encoding='utf8') as file:
        params = yaml.safe_load(file)

    return params


def read_dataframe(settings: Dict) -> pd.DataFrame:
    """
    Read data from disk using specified settings.

    Args:
        settings: Parameters used for input data extraction.

    Returns:
        Dataset in tabular format.
    """
    filepath = settings.pop('filepath')
    logger.info(f'Extract file from {filepath}')
    _, file_extension = os.path.splitext(filepath)

    if file_extension == utils.FileExtension.csv:
        df = pd.read_csv(filepath, **settings)
    elif file_extension in utils.FileExtension.excel:
        df = pd.read_excel(filepath, **settings)
    else:
        raise data_exception.FileExtensionException(file_extension)

    return df


def read_model(settings: Dict) -> ct.SklearnModelTyping:
    """
    Load a trained model from a given path.

    Args:
        settings: Settings containing the path to the file.

    Returns:
        Model object
    """
    filepath = settings['filepath']
    model = joblib.load(filepath)
    return model
