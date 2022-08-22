from typing import Dict, Tuple
import pandas as pd
import os
import yaml
import joblib
from src.exceptions import data as data_exception


class FileExtension:
    """
    This class contains the available files formats to read data.
    """
    csv = 'csv'
    excel = ['xlsx', 'xls']
    # Adding more file extensions here


def read_yaml(file_path: str) -> Dict:
    """
    Given a yaml file path, this function reads content and returns it as a
    Python dict.

    Args:
        file_path (str): yaml's file path.

    Returns:
        a Python dict with file content.
    """
    with open(file_path, encoding='utf8') as file:
        params = yaml.safe_load(file)

    return params


def read_dataframe(settings: Dict) -> pd.DataFrame:
    """
    Read data from disk using specified settings.

    Args:
        settings (Dict): Params used for input data extraction.

    Returns:
        df (pd.DataFrame): the dataset as pandas dataframe.
    """
    filepath = os.path.join(settings['filepath'], settings['data'])
    _, file_extension = os.path.splitext(filepath)
    # extension = settings['data'].split('.')[-1].lower()
    if file_extension == FileExtension.csv:
        # TODO: adding kwargs
        df = pd.read_csv(filepath)
    elif file_extension in FileExtension.excel:
        # TODO: adding kwargs
        df = pd.read_excel(filepath)
    else:
        raise data_exception.FileExtensionException(file_extension)

    return df


def read_model(settings: Dict) -> object:
    """
    Load a trained model from a given path.
    Args:
        settings (Dict): the settings containing the path to the file.
    Returns:
        model (object): The read model from disk.
    """
    filepath = os.path.join(settings['path'], settings['file'])
    model = joblib.load(filepath)
    return model
