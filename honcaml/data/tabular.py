import pandas as pd
from typing import Dict, Tuple

from honcaml.data import base, extract, transform, load
from honcaml.exceptions import data as data_exception
from honcaml.tools import custom_typing as ct
from honcaml.tools.startup import logger


class TabularDataset(base.BaseDataset):
    """
    A dataset consisting in tabular data. The data read come from files
    encoding the data as tables.

    Attributes:
        _features (ct.StrList): Features to be used in dataset.
        _target (ct.StrList): Dataset target column.
        _dataset (pd.DataFrame): Dataset in tabular format.
    """

    def __init__(self) -> None:
        """
        Constructor method of class. It initialises the parameters for this
        specific dataset.
        """
        super().__init__()

        self._features: ct.StrList = []
        self._target: ct.StrList = []

        self._dataset: pd.DataFrame = pd.DataFrame()

    @property
    def features(self) -> ct.StrList:
        """
        Getter method for the '_features' attribute.

        Returns:
            '_features' attribute value.
        """
        return self._features

    @property
    def target(self) -> ct.StrList:
        """
        Getter method for the '_target' attribute.

        Returns:
            '_target' attribute value.
        """
        return self._target

    @property
    def dataframe(self) -> pd.DataFrame:
        """
        Getter method for the '_dataset' attribute.

        Returns:
            '_dataset' attribute value.
        """
        return self._dataset

    @property
    def x(self) -> pd.DataFrame:
        """
        Getter method of features from a dataset.

        Returns:
            Array of x features.
        """
        return self._dataset[self._features]

    @property
    def y(self) -> ct.Array:
        """
        Getter method of target from dataset.

        Returns:
            Array of target.
        """
        if len(self._target) == 0:
            raise data_exception.TargetNotSet()

        y = self._dataset[self._target].values
        if len(self._target) == 1:
            y = y.reshape(-1, 1)
        return y

    @property
    def values(self) -> Tuple[ct.Array, ct.Array]:
        """
        Getter method of features and target from dataset.

        Returns:
            - Array with features.
            - Array with targets.
        """
        if len(self._target) == 0:
            raise data_exception.TargetNotSet()

        x = self._dataset[self._features].values
        y = self._dataset[self._target].values

        return x, y

    def _clean_dataset(
            self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset with previous validation. The validations are the
        following:
            - features and target columns exists.

        Args:
            dataset: Dataset to check.

        Returns:
            Cleaned dataset.
        """
        if self._features:
            try:
                dataset = dataset[self._features + self._target]
            except KeyError as e:
                logger.warning(f'Dataset column features does not exist {e}')
                raise data_exception.ColumnDoesNotExists(f'{self._features}')
        else:
            try:
                self._features = dataset \
                    .drop(columns=self._target).columns.to_list()
            except KeyError as e:
                logger.warning(f'Dataset column features does not exists {e}')
                raise data_exception.ColumnDoesNotExists(f'{self._target}')

        return dataset

    def read(self, settings: Dict) -> None:
        """
        ETL data extract. It follows these steps:
        - Store in a list the features and target.
        - Read data from a file with tabular data (e.g. excel, csv).
        - Clean dataset columns and checks if columns exists.

        Args:
            settings: Input settings for dataset.
        """
        self._features = settings.pop('features', [])
        self._target = settings.pop('target', [])

        dataset = extract.read_dataframe(settings)

        self._dataset = self._clean_dataset(dataset)

    def preprocess(self, settings: Dict):
        """
        ETL data transform. Apply the transformations requested to the data.
        """
        self._dataset = transform.process_data(self._dataset, settings)

    def save(self, settings: Dict):
        """
        ETL data load. Save the dataset into disk.
        """
        load.save_dataframe(self._dataset, settings)
