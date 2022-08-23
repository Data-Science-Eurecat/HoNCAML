from typing import Dict

from src.steps import base
from src.tools import utils


class BenchmarkStep(base.BaseStep):
    """
    The Benchmark steps class is an steps of the main pipeline. The steps
    performs a model ranking by performing a hyperparameter search and model
    selection based on the user and default settings. The extract and load
    methods allow the steps to save and restore executions to/from checkpoints.

    Attributes:
        default_settings (Dict): the default settings for the steps.
        user_settings (Dict): the user defined settings for the steps.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current steps.

        Args:
            default_settings (Dict): the default settings for the steps.
            user_settings (Dict): the user defined settings for the steps.
        """
        super().__init__(default_settings, user_settings)

        # TODO: identify optimizer type (bayesian, random, ...)
        self.optimizer = None
        self.models = []

    def _merge_settings(
            self, default_settings: Dict, user_settings: Dict) -> Dict:
        step_settings = utils.merge_settings(default_settings, user_settings)
        return step_settings

    def extract(self):
        pass

    def transform(self):
        pass

    def load(self):
        pass

    def run(self, objects: Dict) -> Dict:
        """
        TODO
        """
        self.execute()
        objects.update(
            {'model_config': {'library': '', 'hyperparameters': ''}})
        return objects
