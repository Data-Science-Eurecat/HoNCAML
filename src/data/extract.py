from typing import Dict, Tuple
import pandas as pd
import os
import yaml


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


def read_data(settings: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read data from disk using specified settings.

    Args:
        settings (Dict): Params used for input data extraction.

    Returns:
        dataset, target (Tuple[pd.DataFrame, pd.DataFrame]): the dataset and
        the target column.
    """
    filepath = os.path.join(settings['path'], settings['data'])
    extension = settings['data'].split('.')[-1].lower()
    if extension == 'csv':
        df_datatype = pd.read_csv(filepath)
    elif extension in ['xlsx', 'xls']:
        df_datatype = pd.read_excel(filepath)
    else:
        raise Exception(f'File extension {extension} not recognized')
    dataset = df_datatype.drop(settings['target'], axis=1)
    target = df_datatype[[settings['target']]]
    return dataset, target
