from typing import Dict
import pandas as pd
import os
import joblib


def save_dataframe(dataset: pd.DataFrame, settings: Dict) -> None:
    """
    Save the dataset and target columns to disk using the settings given.

    Args:
        dataset (pd.DataFrame): the dataset.
        settings (Dict): Params used for data processing.
    """
    filepath = os.path.join(settings['path'], settings['data'])
    extension = settings['data'].split('.')[-1].lower()
    if extension == 'csv':
        dataset.to_csv(filepath, index=False)
    elif extension in ['xlsx', 'xls']:
        dataset.to_excel(filepath, index=False)
    else:
        raise Exception(f'File extension {extension} not recognized')


def save_model(model: object, settings: Dict) -> None:
    filepath = os.path.join(settings['path'], settings['file'])
    joblib.dump(model, filepath)
