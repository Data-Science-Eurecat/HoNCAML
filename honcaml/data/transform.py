import numpy as np
import pandas as pd
from sklearn import model_selection
from typing import Dict, Tuple

from honcaml.exceptions import data as data_exceptions
from honcaml.tools import custom_typing as ct


def process_data(dataset: pd.DataFrame, settings: Dict) -> pd.DataFrame:
    """
    Preprocess the dataset and target column with settings given.

    Args:
        dataset: Input dataset.
        settings: Parameters used for data processing.

    Returns:
        Processed dataset.
    """
    # TODO: preprocessing logic
    return dataset


# Cross validation utilities

def get_train_test_dataset(
        dataset: ct.Dataset, train_index: np.ndarray,
        test_index: np.ndarray) -> Tuple[ct.Dataset, ct.Dataset]:
    """
    Split the dataset into train and test based on index/position.

    Args:
        dataset: Input dataset.
        train_index: Train indexes.
        test_index: Dataset test indexes.

    Notes:
        Always select the train and test split based on position, for
        this reason when the dataset is a pd.DataFrame or pd.Series
        we use the 'iloc' slicing (numeric position).
        On the other hand, with np.ndarray the normal slicing is fine.

    Returns:
        Train and test datasets.
    """
    if isinstance(dataset, (pd.DataFrame, pd.Series)):
        x_train, x_test = dataset.iloc[train_index], dataset.iloc[test_index]
    elif isinstance(dataset, np.ndarray):
        x_train, x_test = dataset[train_index], dataset[test_index]
    else:
        raise ValueError(
            f'Dataset type {type(dataset)} is not a valid')

    return x_train, x_test


class _CVStrategy:
    """
    Class with the available cross-validation strategies.

    """
    k_fold = 'k_fold'
    repeated_k_fold = 'repeated_k_fold'
    shuffle_split = 'shuffle_split'
    leave_one_out = 'leave_one_out'


class CrossValidationSplit:
    """
    The aim of this class is to wrap different cross-validation strategies
    from sklearn framework.

    Attributes:
        _strategy (str): Cross-validation strategy name.
    """

    def __init__(self, strategy: str) -> None:
        """
        Constructor method. It receives a cross-validation strategy to apply to
        a dataset.

        Args:
            strategy: Cross-validation strategy to apply. The available
                options are the following:
                - k_fold
                - repeated_k_fold
                - shuffle_split
                - leave_one_out
        """
        self._strategy: str = strategy

    @property
    def strategy(self) -> str:
        """
        Getter method for the '_strategy' attribute.

        Returns:
            '_strategy' current value.
        """
        return self._strategy

    def _create_cross_validation_instance(
            self, **kwargs) -> ct.SklearnCrossValidation:
        """
        Creates a new instance of one of the cross-validation strategies
        implemented in sklearn.model_selection. In addition, with *kwargs
        argument, it is possible to pass all the possible parameters
        of the chosen strategy.

        Notes:
            If the cross-validation strategy does not exist, it returns an
            exception.

        Returns:
            A new instance of cross-validation sklearn module.
        """
        if self._strategy == _CVStrategy.k_fold:
            cv_object = model_selection.KFold(**kwargs)
        elif self._strategy == _CVStrategy.repeated_k_fold:
            cv_object = model_selection.RepeatedKFold(**kwargs)
        elif self._strategy == _CVStrategy.shuffle_split:
            cv_object = model_selection.ShuffleSplit(**kwargs)
        elif self._strategy == _CVStrategy.leave_one_out:
            cv_object = model_selection.LeaveOneOut()
        # Adding more strategies here
        else:
            raise data_exceptions.CVStrategyDoesNotExist(self._strategy)

        return cv_object

    def split(
            self, x: ct.Dataset, y: ct.Dataset = None, **kwargs) \
            -> ct.CVGenerator:
        """
        Execute a split method from sklearn cross-validation strategies. In
        addition, the 'kwargs' parameter allows passing additional arguments
        when it creates the object instance. The valid x and y datasets types
        are: pd.DataFrame, pd.Series and np.ndarray.

        Args:
            x: Dataset with features to split.
            y: Dataset with targets to split.

        Yields:
            - Split number
            - Feature array for training
            - Feature array for test
            - Target array for training
            - Target array for test
        """
        cv_object = self._create_cross_validation_instance(**kwargs)

        for split, (train_index, test_index) in \
                enumerate(cv_object.split(x, y), start=1):
            # Get train and test splits for features
            x_train, x_test = get_train_test_dataset(
                x, train_index, test_index)

            # Get train and test splits for targets if it is not None
            if y is not None:
                y_train, y_test = get_train_test_dataset(
                    y, train_index, test_index)
            else:
                y_train, y_test = None, None

            yield split, x_train, x_test, y_train, y_test
