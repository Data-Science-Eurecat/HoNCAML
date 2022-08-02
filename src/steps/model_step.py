from models.regressor_model import RegressorModel
from src.steps.step import Step, StepProcesses
from src.tools import utils
from src.data import extract
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
        super().__init__(default_settings, user_settings)

        # TODO: identify the model type. Assuming RegressorModel for now.
        self.model = RegressorModel()

    def _merge_settings(default_settings: Dict, user_settings: Dict) -> Dict:
        step_settings = utils.merge_settings(default_settings, user_settings)
        return step_settings

    def extract(self):
        self.model.read(self.extract_settings)

    def transform(self):
        # TODO: Train or cross-validate
        # self.model.preprocess(self.transform_settings)
        pass

    def load(self):
        self.model.save(self.load_settings)

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
        # Feed the model with the objects
        super().run()
        # objects.update({})
        return objects
