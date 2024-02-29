import pandas as pd
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from src import processing
from src.frameworks import base

MEMORY_LIMIT = 8000


class AutopytorchClassification(base.BaseTask):
    """
    Class to handle executions for autopytorch classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'f1'

    @staticmethod
    def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Preprocess data for autosklearn regression tasks.

        Args:
            data: Input dataset.

        Returns:
            Processed dataset.
        """
        data = processing.remove_non_numerical_features(data, target)
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
        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values

        self.automl = TabularClassificationTask()

        self.automl.search(
            X_train=X_train,
            y_train=y_train,
            optimize_metric=self.optimize_metric,
            memory_limit=MEMORY_LIMIT,
            total_walltime_limit=parameters['max_seconds'],
            func_eval_time_limit_secs=parameters['max_seconds']/3
        )


class AutopytorchRegression(base.BaseTask):
    """
    Class to handle executions for autopytorch regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'mean_absolute_error'

    @staticmethod
    def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
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

        self.automl = TabularRegressionTask()

        self.automl.search(
            X_train=X_train,
            y_train=y_train,
            optimize_metric=self.optimize_metric,
            memory_limit=MEMORY_LIMIT,
            total_walltime_limit=parameters['max_seconds'],
            func_eval_time_limit_secs=parameters['max_seconds']/3
        )
