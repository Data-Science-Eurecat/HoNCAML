import pandas as pd
from sklearn import model_selection
from typing import Dict, Tuple
from typing_extensions import Protocol

from src.exceptions import data as data_exceptions


def process_data(dataset: pd.DataFrame, target: pd.DataFrame, settings: Dict) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the dataset and target column with settings given.

    Args:
        dataset (pd.DataFrame): the dataset.
        dataset (pd.DataFrame): the target column.
        settings (Dict): Params used for data processing.

    Returns:
        dataset, target (Tuple[pd.DataFrame, pd.DataFrame]): the dataset and
        the target column.
    """
    # TODO: preprocessing logic
    return dataset, target


class SklearnCrossValidationTyping(Protocol):
    def split(self, **kwargs):
        pass


class CVStrategy:
    k_fold = 'k_fold'
    shuffle_split = 'shuffle_split'
    leave_one_out = 'leave_one_out'


class CrossValidationSplit:
    """

    Attributes:
        strategy (str): Cross-validation strategy name
    """

    def __init__(self, strategy: str) -> None:
        """
        Constructor method.

        Args:
            strategy (str): Cross-validation strategy to apply.

        """
        self.strategy = strategy

    def _create_cross_validation_instance(
            self, **kwargs) -> SklearnCrossValidationTyping:
        """
        This function creates a new instance of one of the cross-validation
        strategies implemented in sklearn.model_selection. In addition, with
        kwargs argument, it is possible to pass all the possible parameters
        of the chosen strategy.

        Notes:
            If the cross-validation strategy does not exist, it returns an
            exception.

        Returns:
            a new instance of cross-validation sklearn module.
        """
        if self.strategy == CVStrategy.k_fold:
            cross_validation = model_selection.KFold(**kwargs)
        elif self.strategy == CVStrategy.shuffle_split:
            cross_validation = model_selection.ShuffleSplit(**kwargs)
        elif self.strategy == CVStrategy.leave_one_out:
            cross_validation = model_selection.LeaveOneOut()
        else:
            raise data_exceptions.CVStrategyDoesNotExist(
                self.strategy)

        return cross_validation

    def split(self, x, y=None, **kwargs):
        cv_object = self._create_cross_validation_instance(**kwargs)
