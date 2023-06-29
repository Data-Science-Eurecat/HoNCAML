from typing import Dict

from honcaml.data import tabular, normalization, base as base_dataset
from honcaml.steps import base


class DataStep(base.BaseStep):
    """
    The data step class is a step of the main pipeline. It contains the
    functionalities to perform the ETL on the requested data.

    Attributes:
        _dataset (data.Dataset): Dataset to be handled.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict,
                 global_params: Dict, step_rules: Dict,
                 execution_id: str) -> None:
        """
        Constructor method of class. It initializes the parameters and set up
        the current steps.

        Args:
            default_settings: Default settings for the steps.
            user_settings: User-defined settings for the steps.
            global_params: global parameters for the current pipeline.
            step_rules: Validation rules for this step.
            execution_id: Execution identifier.
        """
        super().__init__(default_settings, user_settings, global_params,
                         step_rules)

        # TODO: identify the dataset type. Assuming TabularDataset for now.
        self._dataset = tabular.TabularDataset()

    @property
    def dataset(self) -> base_dataset.BaseDataset:
        """
        Getter method for the '_dataset' attribute.

        Returns:
            '_dataset' current value.
        """
        return self._dataset

    def _extract(self, settings: Dict) -> None:
        """
        Extract process from the data step ETL. It reads the dataset file
        specified in the configuration.

        Args:
            settings: Settings defining the extract ETL process.
        """
        self._dataset.read(settings)

    def _transform(self, settings: Dict) -> None:
        """
        Transform process from the data step ETL. It prepares the dataset for a
        set of transformations to apply.

        Firstly, it checks if the dataset will be normalized, for this reason
        it creates a new instance of Normalization class as a Dataset class
        attribute.

        Secondly, it runs the basic transformations to a dataset.

        Args:
            settings: Settings defining the transform ETL process.
        """
        # Check normalization settings
        if normalize_settings := settings.pop('normalize', None):
            self._dataset.normalization = normalization.Normalization(
                normalize_settings)

        self._dataset.preprocess(settings)

    def _load(self, settings: Dict) -> None:
        """
        Load process from the data step ETL.

        Args:
            settings: Settings defining the load ETL process.
        """
        self._dataset.save(settings)

    def run(self, metadata: Dict) -> Dict:
        """
        Runs the data steps. Using the dataset created run the ETL functions
        for the specific dataset: extract, transform and load.

        Args:
            metadata: Accumulated pipeline metadata.

        Returns:
            metadata: Updated pipeline metadata with datasat included.
        """
        self.execute()
        metadata.update({'dataset': self._dataset})

        return metadata
