from typing import Dict
import os
import joblib


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
