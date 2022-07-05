from src.tools.step import Step
from typing import Dict


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
        super(default_settings, user_settings)
        self._setup()
        pass
    

    def _setup(self) -> None:
        """
        The function to setup the specific model step.
        """
        # TODO: setup the operations to read, transform and load based on the
        # user and default settings.
        pass


    def extract(self):
        pass

    
    def transform(self):
        pass


    def load(self):
        pass