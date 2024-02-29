import numpy as np
import pandas as pd
from src import processing
from src.frameworks import base

import h2o


class H2oClassification(base.BaseTask):
    """
    Class to handle executions for h2o classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        h2o.init()

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
        h2o_train = h2o.H2OFrame(df_train)
        h2o_train[target] = h2o_train[target].asfactor()

        self.automl = h2o.automl.H2OAutoML(
            max_runtime_secs=parameters['max_seconds'],
            seed=parameters['seed']
        )

        self.automl.train(y=target, training_frame=h2o_train)

    def predict(self, df_test: pd.DataFrame, target: str) -> np.array:
        """
        Predict target given test features.

        Args:
            df_test: Testing dataset.
            target: Target column name.

        Returns:
            y_test: Target array.
        """
        h2o_test = h2o.H2OFrame(df_test.drop(columns=target))
        y_pred = self.automl.leader.predict(
            h2o_test).as_data_frame()['predict']
        return y_pred


class H2oRegression(base.BaseTask):
    """
    Class to handle executions for h2o regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        h2o.init()

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
        h2o_train = h2o.H2OFrame(df_train)
        h2o_train[target] = h2o_train[target].asfactor()

        self.automl = h2o.automl.H2OAutoML(
            max_runtime_secs=parameters['max_seconds'],
            seed=parameters['seed']
        )

        self.automl.train(y=target, training_frame=h2o_train)

    def predict(self, df_test: pd.DataFrame, target: str) -> np.array:
        """
        Predict target given test features.

        Args:
            df_test: Testing dataset.
            target: Target column name.

        Returns:
            y_test: Target array.
        """
        h2o_test = h2o.H2OFrame(df_test.drop(columns=target))
        y_pred = self.automl.leader.predict(
            h2o_test).as_data_frame()['predict']
        return y_pred
