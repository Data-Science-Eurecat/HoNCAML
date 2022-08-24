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
        features ()
        target (ct.): the dataset target column.
        dataset (pd.DataFrame): the dataframe read from the tabular file data.

    """

    def __init__(self) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters for this specific dataset.
        """
        self.features: ct.StrList = []
        self.target: ct.StrList = []

        self.dataset: pd.DataFrame = pd.DataFrame()

    @property
    def x(self) -> ct.Array:
        """
        The 'x' property. This is a getter function for getting x features
        from dataset.

        Returns:
            pd.Dataframe with x features.
        """
        x = self.dataset[self.features] if self.features else self.dataset
        return x.values

    @property
    def y(self) -> ct.Array:
        """
        The 'y' property. This is a getter function for getting target dataset.

        Returns:
            pd.Dataframe with target.
        """
        return self.dataset[self.target].values

    @property
    def values(self) -> Tuple[ct.Array, ct.Array]:
        """
        The 'values' property. This is a getter function for getting values
        from dataset

        Returns:
            two arrays. First one is for x features. Second one is for targets.
        """
        x = self.dataset[self.features] if self.features else self.dataset
        y = self.dataset[self.target]
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
            dataset (pd.DataFrame): cleaned dataframe.
        """
        if self.features:
            try:
                dataset = dataset[self.features + self.target]
            except KeyError as e:
                logger.warning(f'Dataset column features does not exists {e}')
                raise data_exception.ColumnDoesNotExists(f'{self.features}')

        # Check if dataset has a target column/s
        if self.target:
            try:
                dataset[self.target]
            except KeyError as e:
                logger.warning(f'Dataset column features does not exists {e}')
                raise data_exception.ColumnDoesNotExists(f'{self.target}')

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
        self.features = settings.pop('features', [])
        self.target = settings.pop('target', [])

        dataset = extract.read_dataframe(settings)

        self.dataset = self._clean_dataset(dataset)

    def preprocess(self, settings: Dict):
        """
        ETL data transform. Apply the transformations requested to the data.
        """
        self.dataset, self.target = transform.process_data(
            self.dataset, self.target, settings)

    def save(self, settings: Dict):
        """
        ETL data load. Save the dataset into disk.
        """
        dataset = pd.concat((self.dataset, self.target), axis=1)
        load.save_dataframe(dataset, settings)
