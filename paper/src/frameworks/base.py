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
            target: Target column name.

        Returns:
            Processed dataset, if needed.
        """
        return data

    @abstractmethod
    def search_best_model(self) -> None:
        """
        Select best model for the problem at hand.
        Must be implemented by derived classes.
        """
        pass

    def predict(self, df_test: pd.DataFrame, target: str) -> np.array:
        """
        Predict target given test features.

        Args:
            df_test: Testing dataset.
            target: Target column name.

        Returns:
            y_test: Target array.
        """
        X_test = df_test.drop(columns=target).values
        y_pred = self.automl.predict(X_test)
        return y_pred
