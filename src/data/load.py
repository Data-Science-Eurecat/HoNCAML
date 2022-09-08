import joblib
import os
import pandas as pd
import numpy as np
from typing import Dict, List

from src.exceptions import data as data_exception
from src.tools import utils
from src.tools.startup import logger


def save_dataframe(dataset: pd.DataFrame, settings: Dict) -> None:
    """
    Save the dataset and target columns to disk using the settings given.

    Args:
        dataset (pd.DataFrame): the dataset.
        settings (Dict): Params used for data processing.
    """
    filepath = settings.pop('filepath')
    logger.info(f'Load file {filepath}')
    _, file_extension = os.path.splitext(filepath)

    if file_extension == utils.FileExtension.csv:
        dataset.to_csv(filepath, index=False)
    elif file_extension in utils.FileExtension.excel:
        dataset.to_excel(filepath, index=False)
    else:
        raise data_exception.FileExtensionException(file_extension)


def save_model(model: object, settings: Dict) -> None:
    """
    Save a model into disk.

    Args:
        model (object): the model object.
        settings (Dict): parameters to save the model.
    """
    filepath = os.path.join(settings['path'], settings['filename'])
    joblib.dump(model, filepath)


def save_predictions(predictions: List, settings: Dict) -> None:
    """
    Save the list of predictions to disk.

    Args:
        predictions (List): list of predictions to be saved.
        settings (Dict): parameters to save the predictions.
    """
    filename = utils.generate_unique_id('predictions')
    filepath = os.path.join(settings['path'], filename)
    np.save(filepath, predictions)
