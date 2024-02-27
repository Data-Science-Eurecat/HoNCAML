from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseTask(ABC):
    """
    Base class defining an AutoML execution task.

    Attributes:
        automl: Contains the automl object with the best model found.
    """

    def __init__(self) -> None:
        """
        Initialize class settings required attributes.
        """
        self.automl = None
        self.optimize_metric = None

    @staticmethod
    def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Preprocess data for the type of problem, if needed.

        Args:
            data: Input dataset.

        Returns:
            Processed dataset, if needed.
        """
        return data

    @abstractmethod
    def search_best_model(self, X_train: np.array, y_train: np.array,
        parameters: dict):
        """
        Select best model for the problem at hand.
        Must be implemented by derived classes.
        """
        pass

    def predict(self, X_test: np.array) -> np.array:
        """
        Predict target given test features.

        Args:
            X_test: Predict features.

        Returns:
            y_test: Target array.
        """
        y_pred = self.automl.predict(X_test)
        return y_pred
