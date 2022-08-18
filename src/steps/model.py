from typing import Dict

from src.steps import base
from src.tools import utils


class ModelActions:
    fit = 'fit'
    cross_validate = 'cross_validate'
    evaluate = 'evaluate'
    predict = 'predict'


class ModelStep(base.BaseStep):
    """
    The Model step class is an step of the main pipeline. The step performs
    different tasks such as train, predict and evaluate a model. The extract
    and load functions allow the step to save or restore a model.

    Attributes:
        default_settings (Dict): the default settings for the step.
        user_settings (Dict): the user defined settings for the step.
    """

    def validate_step(self):
        pass

    def __init__(self, default_settings: Dict, user_settings: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current step.

        Args:
            default_settings (Dict): the default settings for the step.
            user_settings (Dict): the user defined settings for the step.
        """
        super().__init__(default_settings, user_settings)
        # TODO: get the model config from one of the settings
        self.model_config = self.step_settings.get('model_config', None)

        # TODO: identify the model type. Assuming RegressorModel for now.
        # TODO: split the filename with '.' and it gets the library and
        #  estimator_type
        self.model = RegressorModel()

    def _merge_settings(
            self, default_settings: Dict, user_settings: Dict) -> Dict:
        step_settings = utils.merge_settings(default_settings, user_settings)
        return step_settings

    def extract(self):
        self.model.read(self.extract_settings)

    def transform(self):
        if ModelActions.fit in self.transform_settings:
            self._fit(self.transform_settings['fit'])
        if ModelActions.cross_validate in self.transform_settings:
            self._cross_validate(
                self.transform_settings['cross_validate'])
        if ModelActions.evaluate in self.transform_settings:
            self._evaluate(self.transform_settings['evaluate'])
        if ModelActions.predict in self.transform_settings:
            self._predict(self.transform_settings['predict'])

    def _fit(self):
        pass

    def _cross_validate(self):
        pass

    def _evaluate(self):
        pass

    def _predict(self):
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
        self.model.dataset = objects['dataset']
        if self.model_config is not None:
            model_config = self.model_config
        else:
            model_config = objects['model_config']
        self.model_config = model_config

        self.execute(objects)

        objects.update({'model': self.model})
        return objects
