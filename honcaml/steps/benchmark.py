from typing import Dict

from honcaml.steps import base


class BenchmarkStep(base.BaseStep):
    """
    The benchmark step class is a step of the main pipeline. It performs a
    model ranking by performing a hyperparameter search and model selection
    based on the user and default settings. The extract and load methods allow
    the steps to save and restore executions to/from checkpoints.

    Attributes:
        default_settings (Dict): Default settings for the steps.
        user_settings (Dict): User defined settings for the steps.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict,
                 step_rules: Dict) -> None:
        """
        Constructor method of class. It initializes the parameters and set up
        the current steps.

        Args:
            default_settings: Default settings for the steps.
            user_settings: User-defined settings for the steps.
        """
        super().__init__(default_settings, user_settings, step_rules)

        # TODO: instance the Tuner class with the requested parameters
        self._tuner = None  # tune.Tuner (de ray)
        self._trainables = []  # models.Trainable: one for model to be tuned

    def _extract(self, settings: Dict) -> None:
        """
        Extract process from the benchmark step ETL.

        Args:
            settings: Settings defining the extract ETL process.
        """
        pass

    def _transform(self, settings: Dict) -> None:
        """
        Transform process from the benchmark step ETL.

        Args:
            settings: Settings defining the transform ETL process.
        """
        pass

    def _load(self, settings: Dict) -> None:
        """
        Load process from the benchmark step ETL.

        Args:
            settings: Settings defining the load ETL process.
        """
        pass

    def run(self, metadata: Dict) -> Dict:
        """
        Rusn the benchmark step. Using a benchmark of models run the ETL
        functions to rank them and return the best one.

        Args:
            metadata: Accumulated pipeline metadata.

        Returns:
            metadata: Updated pipeline with the best estimator as a model.
        """
        self.execute()
        metadata.update(
            {'model': 'a model object'})
        return metadata
