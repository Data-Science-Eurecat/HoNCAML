from typing import Dict

from src.data import tabular
from src.steps import base


class DataStep(base.BaseStep):
    """
    The Data steps class is an steps of the main pipeline. It contains the
    functionalities to perform the ETL on the requested data.

    Attributes:
        default_settings (Dict): the default settings for the steps.
        user_settings (Dict): the user defined settings for the steps.
        dataset (data.Dataset): the dataset to be handled.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current steps.

        Args:
            default_settings (Dict): the default settings for the steps.
            user_settings (Dict): the user defined settings for the steps.
        """
        # Getting default settings for data steps
        default_settings = default_settings.get(base.StepType.data)
        # Getting user settings if it exists
        user_settings = user_settings.get(base.StepType.data, {})
        super().__init__(default_settings, user_settings)

        # TODO: identify the dataset type. Assuming TabularDataset for now.
        self.dataset = tabular.TabularDataset()

    def extract(self):
        self.dataset.read(self.extract_settings)

    def transform(self):
        self.dataset.preprocess(self.transform_settings)

    def load(self):
        self.dataset.save(self.load_settings)

    def validate_step(self):
        pass

    def run(self, objects: Dict) -> Dict:
        """
        Run the data steps. Using the dataset created run the ETL functions for
        the specific dataset: extract, transform and load.

        Args:
            objects (Dict): the objects output from each different previous
                steps.

        Returns:
            objects (Dict): the previous objects updated with the ones from
                the current steps: the dataset.
        """
        self.execute()

        objects.update({'dataset': self.dataset})
        return objects
