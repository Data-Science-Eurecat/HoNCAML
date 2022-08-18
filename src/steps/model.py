from typing import Dict, List

from src.steps import base
from src.tools import utils
from src.models import sklearn_model


class ModelActions:
    fit = 'fit'
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

        # TODO: identify the model type. Assuming SklearnModel.
        # TODO: split the filename with '.' and it gets the library and
        #  estimator_type
        self.model = sklearn_model.SklearnModel(
            self.step_settings.pop('estimator_type'))

    def validate_step(self):
        pass

    def _merge_settings(
            self, default_settings: Dict, user_settings: Dict) -> Dict:
        step_settings = utils.merge_settings(default_settings, user_settings)
        return step_settings

    def extract(self):
        self.model.read(self.extract_settings)

    def transform(self):
        if ModelActions.fit in self.transform_settings:
            X, y = self.dataset.get_data()
            self._fit(X, y, self.transform_settings['fit'])
        if ModelActions.predict in self.transform_settings:
            self.predictions = self._predict(
                self.transform_settings['predict'])

    def _fit(self, X, y, settings):
        features = settings.get('features', [])
        if len(features) > 0:
            X = X[features]

        # TODO: implement transformations at dataset level
        X = self.dataset.fit_transform(X)

        params = settings.get('parameters', {})
        self.model.fit(X, y, **params)

    def _predict(self, X, settings) -> List:
        features = settings.get('features', [])
        if len(features) > 0:
            X = X[features]

        # TODO: implement transformations at dataset level
        X = self.dataset.transform(X)

        params = settings.get('parameters', {})
        predictions = self.model.predict(X, **params)
        return predictions

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
        self.dataset = objects['dataset']
        if objects.get('model_config', None) is not None:
            self.model_config = objects['model_config']

        self.execute(objects)

        objects.update({'model': self.model})
        return objects
