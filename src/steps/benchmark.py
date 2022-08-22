from typing import Dict

from src.steps import base
from src.tools import utils


class BenchmarkStep(base.BaseStep):
    """
    The Benchmark step class is an step of the main pipeline. The step
    performs a model ranking by performing a hyperparameter search and model
    selection based on the user and default settings. The extract and load
    methods allow the step to save and restore executions to/from checkpoints.

    Attributes:
        default_settings (Dict): the default settings for the step.
        user_settings (Dict): the user defined settings for the step.
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

        # TODO: identify optimizer type (bayesian, random, ...)
        self.optimizer = None
        self.models = []

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
        The extract process from the benchmark step ETL.
        """
        pass

    def transform(self) -> None:
        """
        The transform process from the benchmark step ETL.
        """
        pass

    def load(self) -> None:
        """
        The load process from the benchmark step ETL.
        """
        pass

    def run(self, objects: Dict) -> Dict:
        """
        Run the benchmark step. Using a benchmark of models run the ETL
        functions to rank them and return the best one.

        Args:
            objects (Dict): the objects output from each different previous
                step.

        Returns:
            objects (Dict): the previous objects updated with the ones from
                the current step: ?.
        """
        self.execute()
        objects.update(
            {'model_config': {'module': '', 'hyperparameters': ''}})
        return objects
