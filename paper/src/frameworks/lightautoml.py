import numpy as np
import pandas as pd
from src import processing
from src.frameworks import base

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

CV = 3


class LightautomlClassification(base.BaseTask):
    """
    Class to handle executions for lightautoml classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()

    @staticmethod
    def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Preprocess data for autosklearn regression tasks.

        Args:
            data: Input dataset.

        Returns:
            Processed dataset.
        """
        if data[target].dtype == 'object':
            data = processing.replace_string_column_to_numeric(
                data, [target])
        return data

    def search_best_model(
            self, X_train: np.array, y_train: np.array,
            parameters: dict) -> None:
        """
        Select best model for the problem at hand and store it within the
        internal `auto_ml` attribute.

        Args:
            X_train: Training features.
            y_train: Training target.
            parameters: General benchmark parameters.
        """
        train_data = pd.DataFrame(X_train)
        train_data['target'] = y_train
        task = Task("binary")
        self.automl = TabularAutoML(
            task=task,
            timeout=parameters['max_seconds'],
            general_params={
                "use_algos": [["linear_l2", "lgb", "cb"]]
            },
            reader_params={"cv": CV, 'random_state': parameters['seed']},
        )
        self.automl.fit_predict(train_data, {'target': 'target'})


class LightautomlRegression(base.BaseTask):
    """
    Class to handle executions for lightautoml regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()

    def search_best_model(
            self, X_train: np.array, y_train: np.array,
            parameters: dict) -> None:
        """
        Select best model for the problem at hand and store it within the
        internal `auto_ml` attribute.

        Args:
            X_train: Training features.
            y_train: Training target.
            parameters: General benchmark parameters.
        """
        train_data = pd.DataFrame(X_train)
        train_data['target'] = y_train
        task = Task("reg")
        self.automl = TabularAutoML(
            task=task,
            timeout=parameters['max_seconds'],
            reader_params={"cv": CV, 'random_state': parameters['seed']},
        )
        self.automl.fit_predict(train_data, {'target': 'target'})
