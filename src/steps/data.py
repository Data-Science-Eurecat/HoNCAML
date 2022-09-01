from typing import Dict

from src.data import tabular, normalization, base as base_dataset
from src.steps import base


class DataStep(base.BaseStep):
    """
    The Data step class is an step of the main pipeline. It contains the
    functionalities to perform the ETL on the requested data.

    Attributes:
        _dataset (data.Dataset): the dataset to be handled.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict,
                 step_rules: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current steps.

        Args:
            default_settings (Dict): the default settings for the steps.
            user_settings (Dict): the user defined settings for the steps.
        """
        super().__init__(default_settings, user_settings, step_rules)

        # TODO: identify the dataset type. Assuming TabularDataset for now.
        self._dataset = tabular.TabularDataset()

    @property
    def dataset(self) -> base_dataset.BaseDataset:
        """
        This is a getter method. This function returns the '_dataset'
        attribute.

        Returns:
            (base_dataset.BaseDataset): dataset instance.
        """
        return self._dataset

    def _extract(self, settings: Dict) -> None:
        """
        The extract process from the data step ETL. This function reads the
        dataset file specified in the settings dict.

        Args:
            settings (Dict): the settings defining the extract ETL process.
        """
        self._dataset.read(settings)

    def _transform(self, settings: Dict) -> None:
        """
        The transform process from the data step ETL. This function prepare
        dataset for a set of transformations to apply.

        Firstly, it checks if the dataset will be normalized, for this reason
        it creates a new instance of Normalization class as a Dataset class
        attribute.

        Secondly, it runs the basic transformations to a dataset.

        Args:
            settings (Dict): the settings defining the transform ETL process.
        """
        # Check normalization settings
        if normalize_settings := settings.pop('normalize', None):
            self._dataset.normalization = normalization.Normalization(
                normalize_settings)

        self._dataset.preprocess(settings)

    def _load(self, settings: Dict) -> None:
        """
        The load process from the data step ETL.

        Args:
            settings (Dict): the settings defining the load ETL process.
        """
        self._dataset.save(settings)

    def run(self, metadata: Dict) -> Dict:
        """
        Run the data steps. Using the dataset created run the ETL functions for
        the specific dataset: extract, transform and load.

        Args:
            metadata (Dict): the objects output from each different previous
                steps.

        Returns:
            metadata (Dict): the previous objects updated with the ones from
                the current steps: the dataset.
        """
        self.execute()
        metadata.update({'dataset': self._dataset})

        return metadata
