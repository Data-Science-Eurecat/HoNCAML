from typing import Dict, Tuple
import pandas as pd


def process_data(dataset: pd.DataFrame, target: pd.DataFrame, settings: Dict)\
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the dataset and target column with settings given.

    Args:
        dataset (pd.DataFrame): the dataset.
        dataset (pd.DataFrame): the target column.
        settings (Dict): Params used for data processing.

    Returns:
        dataset, target (Tuple[pd.DataFrame, pd.DataFrame]): the dataset and
        the target column.
    """
    # TODO: preprocessing logic
    return dataset, target
