from typing import Dict

from honcaml.steps import base
from honcaml.tools import utils


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

        # TODO: instance the Tuner class with the requested parameters
        self._tuner = None  # tune.Tuner (de ray)
        self._trainables = []  # models.Trainable: one for each model to be tuned

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
