from typing import Dict

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


def read_data(settings: Dict) -> Dict:
    """
    Read data from disk using specified settings.

    Args:
        settings (Dict): Params used for input data extraction.

    Returns:
        data (Dict): datasets for each specified data type in settings
    """
    data = {}
    data_types = settings['data']
    for data_type, type_filename in data_types.items():
        data[data_type] = load_datatype(settings['path'], type_filename)
    return data
