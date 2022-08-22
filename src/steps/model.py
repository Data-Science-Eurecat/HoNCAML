from typing import Dict, List

from src.steps import base
from src.tools import utils
from src.models import sklearn_model, general
from src.data import transform


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
        The extract process from the model step ETL.
        """
        self.model.read(self.extract_settings)

    def transform(self) -> None:
        """
        The transform process from the model step ETL.
        """
        if self.model.estimator is None:
            self.model.build_model(
                self.model_config, self.dataset.normalizations)
        if ModelActions.fit in self.transform_settings:
            self._fit(self.transform_settings['fit'])
        if ModelActions.predict in self.transform_settings:
            self.predictions = self._predict(
                self.transform_settings['predict'])

    def _fit(self, settings: Dict) -> None:
        """
        The training function for the model step. It performs a training step
        on the whole dataset and a cross-validation one if specified.

        Args:
            settings (Dict): the training and cross-validation configuration.
        """
        X, y = self.dataset.get_data()
        if settings.get('cross_validation', None) is not None:
            # Run the cross-validation
            cv_split = transform.CrossValidationSplit(
                settings['cross_validation'].pop('strategy'))

            results = []
            for split, X_train, X_test, y_train, y_test in \
                    cv_split.split(X, y, settings.pop('cross_validation')):
                # Afegir normalizations
                self.model.fit(X_train, y_train, **settings)

                results.append(self.model.evaluate(X_test, y_test, **settings))
        # Group cv metrics
        self.cv_results = general.aggregate_cv_results(results)
        # Train the model with whole data
        self.model.fit(X, y)

    def _predict(self, settings: Dict) -> List:
        """
        The predict function for the model step. It performs predictions for
        the whole dataset by using the model.

        Args:
            settings (Dict): the predict configuration.

        Returns:
            predictions (List): the prediction for each sample.
        """
        X, y = self.dataset.get_data()
        predictions = self.model.predict(X, **settings)
        return predictions

    def load(self) -> None:
        """
        The load process from the model step ETL.
        """
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
