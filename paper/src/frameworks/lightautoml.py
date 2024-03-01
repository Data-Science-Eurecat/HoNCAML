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
            data = processing.replace_string_columns_to_numeric(
                data, [target])
        return data

    def search_best_model(
            self, df_train: pd.DataFrame, target: str,
            parameters: dict) -> None:
        """
        Select best model for the problem at hand and store it within the
        internal `auto_ml` attribute.

        Args:
            df_train: Training dataset.
            target: Target column name.
            parameters: General benchmark parameters.
        """
        task = Task("binary")
        self.automl = TabularAutoML(
            task=task,
            timeout=parameters['max_seconds'],
            general_params={
                "use_algos": [["linear_l2", "lgb", "cb"]]
            },
            reader_params={"cv": CV, 'random_state': parameters['seed']},
        )
        self.automl.fit_predict(df_train, {'target': target})

    def predict(self, df_test: pd.DataFrame, target: str) -> np.array:
        """
        Predict target given test features.

        Args:
            df_test: Testing dataset.
            target: Target column name.

        Returns:
            y_test: Target array.
        """
        y_pred = self.automl.predict(df_test).data[:, 0]
        return y_pred


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
            self, df_train: pd.DataFrame, target: str,
            parameters: dict) -> None:
        """
        Select best model for the problem at hand and store it within the
        internal `auto_ml` attribute.

        Args:
            df_train: Training dataset.
            target: Target column name.
            parameters: General benchmark parameters.
        """
        task = Task("reg")
        self.automl = TabularAutoML(
            task=task,
            timeout=parameters['max_seconds'],
            reader_params={"cv": CV, 'random_state': parameters['seed']},
        )
        self.automl.fit_predict(df_train, {'target': target})

    def predict(self, df_test: pd.DataFrame, target: str) -> np.array:
        """
        Predict target given test features.

        Args:
            df_test: Testing dataset.
            target: Target column name.

        Returns:
            y_test: Target array.
        """
        y_pred = self.automl.predict(df_test).data[:, 0]
        return y_pred
