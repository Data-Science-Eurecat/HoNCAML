import pandas as pd
from src import processing
from src.frameworks import base

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import make_scorer
from tpot import TPOTClassifier, TPOTRegressor


class TpotClassification(base.BaseTask):
    """
    Class to handle executions for flaml classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'f1_macro'

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

        self.automl = TPOTClassifier(
            random_state=parameters['seed'],
            scoring=self.optimize_metric,
            max_time_mins=parameters['max_seconds'] / 60
        )
        self.automl.fit(X_train, y_train)


class TpotRegression(base.BaseTask):
    """
    Class to handle executions for autosklearn regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'f1_macro'

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

    @staticmethod
    def my_custom_MAPE(y_true, y_pred):
        return mean_absolute_percentage_error(y_true, y_pred)

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

        my_custom_scorer = make_scorer(
            self.my_custom_MAPE, greater_is_better=False)
        self.automl = TPOTRegressor(
            random_state=parameters['seed'],
            scoring=my_custom_scorer,
            max_time_mins=parameters['max_seconds'] / 60
        )
        self.automl.fit(X_train, y_train)
