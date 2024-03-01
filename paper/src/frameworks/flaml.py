import pandas as pd
from src.frameworks import base

from flaml import AutoML


class FlamlClassification(base.BaseTask):
    """
    Class to handle executions for flaml classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.problem_type = 'classification'
        self.optimize_metric = 'macro_f1'

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

        automl_settings = {
            "time_budget": parameters['max_seconds'],
            "metric": self.optimize_metric,
            "task": self.problem_type,
            "log_file_name": "flaml.log",
            "seed": parameters['seed']
        }
        self.automl = AutoML()
        self.automl.fit(X_train, y_train, **automl_settings)


class FlamlRegression(base.BaseTask):
    """
    Class to handle executions for flaml regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.problem_type = 'regression'
        self.optimize_metric = 'mape'

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

        automl_settings = {
            "time_budget": parameters['max_seconds'],
            "metric": self.optimize_metric,
            "task": self.problem_type,
            "log_file_name": "flaml.log",
            "seed": parameters['seed']
        }
        self.automl = AutoML()
        self.automl.fit(X_train, y_train, **automl_settings)
