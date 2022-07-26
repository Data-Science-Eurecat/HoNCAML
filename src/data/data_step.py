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
                self.default_settings['phases'][task],
                self.user_settings[task])
        # TODO: identify the dataset type. Assuming TabularDataset for now.
        self.dataset = TabularDataset(action_settings)

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
        self.dataset.extract()
        self.dataset.transform()
        self.dataset.load()
        objects.update({'dataset': self.dataset})
        return objects
