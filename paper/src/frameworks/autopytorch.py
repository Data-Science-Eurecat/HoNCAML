import numpy as np
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.api.tabular_regression import TabularRegressionTask
from src.frameworks import base

MEMORY_LIMIT = 8000


class AutosklearnClassification(base.BaseTask):
    """
    Class to handle executions for autopytorch classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'f1'

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
        self.automl = TabularClassificationTask()

        self.automl.search(
            X_train=X_train,
            y_train=y_train,
            optimize_metric=self.optimize_metric,
            memory_limit=MEMORY_LIMIT,
            total_walltime_limit=parameters['max_time'],
            func_eval_time_limit_secs=parameters['max_time']/3
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
        self.automl = TabularRegressionTask()

        self.automl.search(
            X_train=X_train,
            y_train=y_train,
            optimize_metric=self.optimize_metric,
            memory_limit=MEMORY_LIMIT,
            total_walltime_limit=parameters['max_time'],
            func_eval_time_limit_secs=parameters['max_time']/3
        )
