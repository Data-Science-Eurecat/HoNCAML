import pandas as pd
import keras_tuner
from keras import backend as K
from src.frameworks import base

import autokeras as ak

EPOCHS = 10
MAX_TRIALS = 3


class AutokerasClassification(base.BaseTask):
    """
    Class to handle executions for autokeras classification tasks.
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
        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values

        self.automl = ak.StructuredDataClassifier(
            overwrite=True, max_trials=MAX_TRIALS,
        )

        self.automl.fit(X_train, y_train, epochs=EPOCHS)


class AutokerasRegression(base.BaseTask):
    """
    Class to handle executions for autokeras regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()

        def mae(y_true, y_pred):
            mae = K.mean(K.abs(y_true - y_pred))
            return mae

        self.optimize_metric = mae

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

        self.automl = ak.StructuredDataRegressor(
            overwrite=True, max_trials=MAX_TRIALS,
            metrics=[self.optimize_metric],
            objective=keras_tuner.Objective('val_mae', direction='min')
        )

        self.automl.fit(X_train, y_train, epochs=EPOCHS)
