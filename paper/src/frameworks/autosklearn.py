import pandas as pd
from src import processing
from src.frameworks import base

import autosklearn.classification
import autosklearn.metrics
import autosklearn.regression


class AutosklearnClassification(base.BaseTask):
    """
    Class to handle executions for autosklearn classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = autosklearn.metrics.f1_macro

    @staticmethod
    def preprocess_data(
            data: pd.DataFrame, target: str, *args) -> pd.DataFrame:
        """
        Preprocess data for autosklearn regression tasks.

        Args:
            data: Input dataset.

        Returns:
            Processed dataset.
        """
        data = processing.remove_non_numerical_features(data, target)
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
        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values

        self.automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=parameters['max_seconds'],
            seed=parameters['seed'],
            metric=self.optimize_metric)
        self.automl.fit(X_train, y_train)


class AutosklearnRegression(base.BaseTask):
    """
    Class to handle executions for autosklearn regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = autosklearn.metrics.mean_absolute_error

    @staticmethod
    def preprocess_data(
            data: pd.DataFrame, target: str, *args) -> pd.DataFrame:
        """
        Preprocess data for autosklearn regression tasks.

        Args:
            data: Input dataset.

        Returns:
            Processed dataset.
        """
        data = processing.remove_non_numerical_features(data, target)
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
        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values

        self.automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=parameters['max_seconds'],
            seed=parameters['seed'],
            metric=self.optimize_metric)
        self.automl.fit(X_train, y_train)
