from typing import Dict

from src.data import tabular
from src.steps import base
from src.tools import utils


class DataStep(base.BaseStep):
    """
    The Data step class is an step of the main pipeline. It contains the
    functionalities to perform the ETL on the requested data.

    Attributes:
        default_settings (Dict): the default settings for the step.
        user_settings (Dict): the user defined settings for the step.
        dataset (data.Dataset): the dataset to be handled.
    """

    def validate_step(self):
        pass

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current step.

        Args:
            default_settings (Dict): the default settings for the step.
            user_settings (Dict): the user defined settings for the step.
        """
        super().__init__(default_settings, user_settings)

        # TODO: identify the dataset type. Assuming TabularDataset for now.
        self.dataset = tabular.TabularDataset()

    def _merge_settings(
            self, default_settings: Dict, user_settings: Dict) -> Dict:
        """
        Merge the user defined settings with the default ones.

        Args:
            default_settings (Dict): the default settings for the step.
            default_settings (Dict): the user defined settings for the step.

        Returns:
            merged_settings (Dict): the user and default settings merged.
        """
        step_settings = utils.merge_settings(default_settings, user_settings)
        return step_settings

    def validate_step(self) -> None:
        """
        Validates the settings for the step ensuring that the step has the
        mandatory keys to run.
        """
        pass

    def extract(self) -> None:
        """
        The extract process from the data step ETL.
        """
        self.dataset.read(self.extract_settings)

    def transform(self) -> None:
        """
        The transform process from the data step ETL.
        """
        self.dataset.preprocess(self.transform_settings)

    def load(self) -> None:
        """
        The load process from the data step ETL.
        """
        self.dataset.save(self.load_settings)

    def run(self, objects: Dict) -> Dict:
        """
        Run the data step. Using the dataset created run the ETL functions for
        the specific dataset: extract, transform and load.

        Args:
            objects (Dict): the objects output from each different previous
                step.

        Returns:
            objects (Dict): the previous objects updated with the ones from
                the current step: the dataset.
        """
        self.execute()

        objects.update({'dataset': self.dataset})
        return objects
