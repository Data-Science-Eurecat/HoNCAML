from typing import Dict, Tuple
import pandas as pd
import os
import yaml
import joblib


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
        df_datatype (pd.DataFrame): the dataset as pandas dataframe.
    """
    filepath = os.path.join(settings['path'], settings['data'])
    extension = settings['data'].split('.')[-1].lower()
    if extension == 'csv':
        df_datatype = pd.read_csv(filepath)
    elif extension in ['xlsx', 'xls']:
        df_datatype = pd.read_excel(filepath)
    else:
        raise Exception(f'File extension {extension} not recognized')
    return df_datatype


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
