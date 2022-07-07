from data.tabular_dataset import TabularDataset
from src.tools.step import Step
from src.tools import utils
from typing import Dict


class DataStep(Step):
    """
    The Data step class is an step of the main pipeline. It contains the
    functionalities to perform the ETL on the requested data.

    Attributes:
        default_settings (Dict): the default settings for the step.
        user_settings (Dict): the user defined settings for the step.
        dataset (data.Dataset): the dataset to be handled.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current step.

        Args:
            default_settings (Dict): the default settings for the step.
            user_settings (Dict): the user defined settings for the step.
        """
        super().__init__(default_settings, user_settings)
        self._setup()

    def _setup(self) -> None:
        """
        The function to setup the specific data step.
        """
        action_settings = {}
        for task in self.user_settings:
            action_settings[task] = utils.merge_settings(
                self.default_settings[task], self.user_settings[task])
        # TODO: identify the dataset type. Assuming TabularDataset for now.
        self.dataset = TabularDataset(action_settings)

    def extract(self):
        """
        ETL data extract consisting in reading the dataset.
        """
        self.dataset.extract()

    def transform(self):
        """
        ETL data transform consisting in applying transformations to the data,
        data processing.
        """
        self.dataset.transform()

    def load(self):
        """
        ETL data load consisting in saving the dataset.
        """
        self.dataset.load()
