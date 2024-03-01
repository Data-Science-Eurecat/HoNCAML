import numpy as np
import pandas as pd
from src.frameworks import base

from autogluon.tabular import TabularPredictor


class AutogluonClassification(base.BaseTask):
    """
    Class to handle executions for Autogluon classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'f1_macro'
        self.problem_type = 'binary'

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
        self.automl = TabularPredictor(
            label=target, problem_type=self.problem_type,
            eval_metric=self.optimize_metric)

        self.automl.fit(
            df_train, time_limit=parameters['max_seconds'])

    def predict(self, df_test: pd.DataFrame, target: str) -> np.array:
        """
        Predict target given test features.

        Args:
            df_test: Testing dataset.
            target: Target column name.

        Returns:
            y_test: Target array.
        """
        y_pred = self.automl.predict(df_test)
        return y_pred


class AutogluonRegression(base.BaseTask):
    """
    Class to handle executions for Autogluon regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'mae'
        self.problem_type = 'regression'

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
        self.automl = TabularPredictor(
            label=target, problem_type=self.problem_type,
            eval_metric=self.optimize_metric)

        self.automl.fit(
            df_train, time_limit=parameters['max_seconds'])

    def predict(self, df_test: pd.DataFrame, target: str) -> np.array:
        """
        Predict target given test features.

        Args:
            df_test: Testing dataset.
            target: Target column name.

        Returns:
            y_test: Target array.
        """
        y_pred = self.automl.predict(df_test)
        return y_pred
