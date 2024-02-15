import numpy as np
import pandas as pd
from typing import Dict, Tuple

from honcaml.exceptions import data as data_exceptions
from honcaml.tools import utils, custom_typing as ct
from sklearn.preprocessing import OneHotEncoder


def process_data(
        dataset: pd.DataFrame, target: str, settings: Dict) -> pd.DataFrame:
    """
    Preprocess the dataset and target column with settings given.

    Args:
        dataset: Input dataset.
        settings: Parameters used for data processing.

    Returns:
        Processed dataset.
    """
    do = True
    if 'encoding' in settings.keys():
        if 'OHE' in settings['encoding']:
            if settings['encoding']['OHE'] is False:
                do = False

    features = []
    if do is True:
        if 'encoding' in settings.keys():
            if 'features' in settings['encoding']:
                features = features + utils.ensure_input_list(
                    settings['encoding']['features'])
            else:
                features = dataset.columns

        drop_col = []
        for col in features:
            if col != target:
                if dataset[col].dtype == 'object':
                    if len(dataset[col].unique()) > settings[
                            'encoding']['max_values']:
                        drop_col.append(col)
                    else:
                        encoder = OneHotEncoder()
                        data = encoder.fit_transform(
                            dataset[col].values.reshape(-1, 1)).toarray()
                        name_col = col+'_'+encoder.categories_[0]
                        dataset = pd.concat([
                            dataset.drop(col, axis=1),
                            pd.DataFrame(data, columns=name_col)
                        ], axis=1)

        dataset = dataset.drop(drop_col, axis=1)
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


class CrossValidationSplit:
    """
    The aim of this class is to wrap possible cross-validation classes from
    sklearn framework.

    Attributes:
        _module (str): Cross-validation module.
        _data (Dict): Dict with additional parameters to pass to
            cross validation module.
    """

    def __init__(self, module: str, params: Dict = None) -> None:
        """
        Constructor method. It receives a cross-validation module and
        parameters to apply to the dataset.

        Args:
            module: Cross-validation module to apply.
            params: Additional parametes to pass to module.
        """
        self._module: str = module
        self._params: Dict = params

    def _create_cross_validation_instance(self) -> ct.SklearnCrossValidation:
        """
        Creates a new instance of cross validation. In addition, with *kwargs
        argument, it is possible to pass all the possible parameters
        of the chosen module.

        Notes:
            If the cross-validation module does not exist, it returns an
            exception.

        Returns:
            A new instance of cross-validation module.
        """
        try:
            cv_object = utils.import_library(self._module, self._params)
        except AttributeError:
            raise data_exceptions.CVModuleDoesNotExist(self._module)

        return cv_object

    def split(
            self, x: ct.Dataset, y: ct.Dataset = None) -> ct.CVGenerator:
        """
        Execute a split method from cross-validation module. In addition, the
        'kwargs'  parameter allows passing additional arguments when it creates
        the object instance. The valid x and y datasets types are:
        pd.DataFrame, pd.Series and np.ndarray.

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
        cv_object = self._create_cross_validation_instance()

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
