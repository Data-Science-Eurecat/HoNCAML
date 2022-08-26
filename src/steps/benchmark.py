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
        self._optimizer = None
        self._models = []

    def _validate_step(self) -> None:
        """
        Validates the settings for the step ensuring that the step has the
        mandatory keys to run.
        """
        pass

    def _extract(self, settings: Dict) -> None:
        """
        The extract process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the extract ETL process.
        """
        pass

    def _transform(self, settings: Dict) -> None:
        """
        The transform process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the transform ETL process.
        """
        pass

    def _load(self, settings: Dict) -> None:
        """
        The load process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the load ETL process.
        """
        pass

    def run(self, metadata: Dict) -> Dict:
        """
        Run the benchmark step. Using a benchmark of models run the ETL
        functions to rank them and return the best one.

        Args:
            metadata (Dict): the objects output from each different previous
                step.

        Returns:
            metadata (Dict): the previous objects updated with the ones from
                the current step: the best estimator as a model from this
                library.
        """
        self.execute()
        metadata.update(
            {'model': 'a model objecct'})
        return metadata
