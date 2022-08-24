import pandas as pd
from typing import Dict, Tuple

from src.data import base, extract, transform, load
from src.exceptions import data as data_exception
from src.tools import custom_typing as ct
from src.tools.startup import logger


class TabularDataset(base.BaseDataset):
    """
    A dataset consisting in tabular data. The data read come from files
    encoding the data as tables.

    Attributes:
        action_settings (Dict): the parameters that define each action from 
        the ETL process.
        _features ()
        _target (ct.): the dataset target column.
        _dataset (pd.DataFrame): the dataframe read from the tabular file data.

    """

    def __init__(self) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters for this specific dataset.
        """
        self._features: ct.StrList = []
        self._target: ct.StrList = []

        self._dataset: pd.DataFrame = pd.DataFrame()

    @property
    def features(self) -> ct.StrList:
        """
        This is a getter method. This function returns the '_features'
        attribute.

        Returns:
            (List[str]): list of features columns.
        """
        return self._features

    @property
    def target(self) -> ct.StrList:
        """
        This is a getter method. This function returns the '_target'
        attribute.

        Returns:
            (List[str]): list of target columns
        """
        return self._target

    @property
    def dataset(self) -> pd.DataFrame:
        """
        This is a getter method. This function returns the '_dataset'
        attribute.

        Returns:
            (pd.DataFrame): dataset as pd.DataFrame
        """
        return self._dataset

    @property
    def x(self) -> ct.Array:
        """
        This is a getter method. This function returns the x features
        from dataset.

        Returns:
            (pd.Dataframe): pd.DataFrame with x features.
        """
        x = self._dataset[self._features] if self._features else self._dataset
        return x.values

    @property
    def y(self) -> ct.Array:
        """
        This is a getter method. This function returns the target from dataset.

        Returns:
            (pd.Dataframe): pd.DataFrame with target.
        """
        return self._dataset[self._target].values

    @property
    def values(self) -> Tuple[ct.Array, ct.Array]:
        """
        This is a getter method. This function returns the x and y values from
        dataset as np.ndarray.

        Returns:
            (Tuple[np.ndarray], Tuple[np.ndarray]): First array contains x
            feature values. Second one contains targets.
        """
        x = self._dataset[self._features] if self._features else self._dataset
        y = self._dataset[self._target]
        return x.values, y.values

    def _clean_dataset(
            self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe, this function runs a set of validations, and
        selected the columns.
        The validations are the following:
            - features and target columns exists

        Args:
            dataset (pd.DataFrame): dataframe to clean.

        Returns:
            (pd.DataFrame): cleaned dataframe.
        """
        if self._features:
            try:
                dataset = dataset[self._features + self._target]
            except KeyError as e:
                logger.warning(f'Dataset column features does not exists {e}')
                raise data_exception.ColumnDoesNotExists(f'{self._features}')

        # Check if dataset has a target column/s
        if self._target:
            try:
                dataset[self._target]
            except KeyError as e:
                logger.warning(f'Dataset column features does not exists {e}')
                raise data_exception.ColumnDoesNotExists(f'{self._target}')

        return dataset

    def read(self, settings: Dict) -> None:
        """
        ETL data extract. Firstly, it stores in a list the features and target.
        Then it reads data from a file that encodes the data as tables
        (e.g. excel, csv). Finally, it cleans dataframes columns and checks
        if columns exists.

        Args:
            settings (Dict): a dict that contains settings.
        """
        self._features = settings.pop('features', [])
        self._target = settings.pop('target', [])

        dataset = extract.read_dataframe(settings)

        self._dataset = self._clean_dataset(dataset)

    def preprocess(self, settings: Dict):
        """
        ETL data transform. Apply the transformations requested to the data.
        """
        self._dataset, self._target = transform.process_data(
            self._dataset, self._target, settings)

    def save(self, settings: Dict):
        """
        ETL data load. Save the dataset into disk.
        """
        dataset = pd.concat((self._dataset, self._target), axis=1)
        load.save_dataframe(dataset, settings)
