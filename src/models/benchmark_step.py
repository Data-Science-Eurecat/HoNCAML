from src.tools.step import Step
from typing import Dict


class BenchmarkStep(Step):
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
        self._setup()
        pass

    def _setup(self) -> None:
        """
        The function to setup the specific benchmark step.
        """
        # TODO: setup the operations to read, transform and load based on the
        # user and default settings.
        pass

    def run(self) -> None:
        """
        TODO
        """
        pass
