from typing import Dict
import pandas as pd
import os


def save_data(dataset: pd.DataFrame, target: pd.DataFrame, settings: Dict)\
        -> None:
    """
    Save the dataset and target columns to disk using the settings given.

    Args:
        dataset (pd.DataFrame): the dataset.
        dataset (pd.DataFrame): the target column.
        settings (Dict): Params used for data processing.
    """
    df = pd.concat((dataset, target), axis=1)
    filepath = os.path.join(settings['path'], settings['data'])
    extension = settings['data'].split('.')[-1].lower()
    if extension == 'csv':
        df.to_csv(filepath, index=False)
    elif extension in ['xlsx', 'xls']:
        df.to_excel(filepath, index=False)
    else:
        raise Exception(f'File extension {extension} not recognized')
