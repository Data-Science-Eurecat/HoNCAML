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
