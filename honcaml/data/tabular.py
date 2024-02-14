import pandas as pd
from typing import Dict, Tuple

from honcaml.data import base, extract, transform, load
from honcaml.exceptions import data as data_exception
from honcaml.steps.model import ModelActions
from honcaml.tools.startup import logger
from honcaml.tools import custom_typing as ct


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

    def _clean_dataset_for_model(
            self, dataset: pd.DataFrame, model_actions: str) -> pd.DataFrame:
        """
        Clean dataset and perform validations to ensure that model will work.
        What is done is the following, in this order:
        1. Features are defined if not specified in the settings, depending on
           dataset columns.
        2. Validations are performed:
            - For fit: features and target columns exist.
            - For predictions: features exist.
        3. All non-required columns are removed from the dataset.

        Args:
            dataset: Dataset to check.
            model_actions: Model actions to be performed.

        Returns:
            Cleaned dataset.
        """
        # Set features if not specified
        ds_cols = list(dataset.columns)
        if not self._features:
            if ModelActions.fit in model_actions and self._target in ds_cols:
                self._features = list(
                    dataset.drop(columns=self._target).columns)
            else:
                self._features = ds_cols
        # Perform validations
        feat_inters = set(self._features).intersection(set(ds_cols))
        if not feat_inters:
            raise data_exception.ColumnDoesNotExists(f'{self._features}')
        if ModelActions.fit in model_actions:
            if self._target not in ds_cols:
                raise data_exception.ColumnDoesNotExists(f'{self._target}')
        # Remove all columns not features and not target
        cols_to_rm = set(ds_cols).difference(
            set(self._features + [self._target]))
        logger.debug(f'Columns to remove from dataset: {cols_to_rm}')
        dataset = dataset.drop(columns=cols_to_rm)
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
        self._dataset = extract.read_dataframe(settings)

    def preprocess(self, settings: Dict):
        """
        ETL data transform. Apply the transformations requested to the data.
        """
        dataset = transform.process_data(self._dataset, self.target, settings)
        self._features = settings.pop('features', [])
        model_actions = [ModelActions.fit]
        if len(dataset) != 0:
            self._dataset = self._clean_dataset_for_model(dataset,
                                                          model_actions)

    def save(self, settings: Dict):
        """
        ETL data load. Save the dataset into disk.
        """
        load.save_dataframe(self._dataset, settings)
