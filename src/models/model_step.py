from models.regressor_model import RegressorModel
from src.tools.step import Step
from typing import Dict
from src.tools import utils


class ModelStep(Step):
    """
    The Model step class is an step of the main pipeline. The step performs
    different tasks such as train, predict and evaluate a model. The extract
    and load functions allow the step to save or restore a model.

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

    def _setup(self) -> None:
        """
        The function to setup the specific model step.
        """
        action_settings = {}
        for task in self.user_settings:
            action_settings[task] = utils.merge_settings(
                self.default_settings['phases'][task],
                self.user_settings[task])
        # TODO: identify the model type. Assuming RegressorModel for now.
        self.model = RegressorModel(action_settings)

    def run(self, objects: Dict) -> None:
        """
        Run the model step. Using the model created run the ETL functions for
        the specific model: extract, transform and load.

        Args:
            objects (Dict): the objects output from each different previous
                step.

        Returns:
            objects (Dict): the previous objects updated with the ones from
                the current step: ?.
        """
        self.model.extract()
        self.model.transform()
        self.model.load()
        objects.update({})
        return objects
