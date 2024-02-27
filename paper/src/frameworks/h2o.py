import h2o
import numpy as np
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
    def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Preprocess data for autosklearn regression tasks.

        Args:
            data: Input dataset.

        Returns:
            Processed dataset.
        """
        if data[target].dtype == 'object':
            data[target] = processing.replace_string_column_to_numeric(
                data[target])
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
        h2o_train = h2o.H2OFrame(train_data)
        h2o_train['target'] = h2o_train['target'].asfactor()

        self.automl = h2o.automl.H2OAutoML(
            max_runtime_secs=parameters['max_seconds'],
            seed=parameters['seed']
        )

        self.automl.train(y='target', training_frame=h2o_train)

    def predict(self, X_test: np.array) -> np.array:
        """
        Predict target given test features.

        Args:
            X_test: Predict features.

        Returns:
            y_test: Target array.
        """
        test_data = pd.DataFrame(X_test)
        h2o_test = h2o.H2OFrame(test_data)
        y_pred = self.automl.leader.predict(
            h2o_test).as_data_frame()['predict']
        return y_pred


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
        h2o_train = h2o.H2OFrame(train_data)
        h2o_train['target'] = h2o_train['target'].asfactor()

        self.automl = h2o.automl.H2OAutoML(
            max_runtime_secs=parameters['max_seconds'],
            seed=parameters['seed']
        )

        self.automl.train(y='target', training_frame=h2o_train)

    def predict(self, X_test: np.array) -> np.array:
        """
        Predict target given test features.

        Args:
            X_test: Predict features.

        Returns:
            y_test: Target array.
        """
        test_data = pd.DataFrame(X_test)
        h2o_test = h2o.H2OFrame(test_data)
        y_pred = self.automl.leader.predict(
            h2o_test).as_data_frame()['predict']
        return y_pred
