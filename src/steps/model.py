from typing import Dict, List

from src.steps import base
from src.models import base as base_model
from src.models import sklearn_model, general
from src.data import transform
from src.exceptions import model as model_exceptions


class ModelActions:
    fit = 'fit'
    predict = 'predict'


default_estimator = {
    'module': 'sklearn.ensemble.RandomForestClassifier',
    'hyperparameters': {}
}


class ModelStep(base.BaseStep):
    """
    The Model steps class is an steps of the main pipeline. The steps performs
    different tasks such as train, predict and evaluate a model. The extract
    and load functions allow the steps to save or restore a model.

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
        self.estimator_type = user_settings.pop('estimator_type')
        self.estimator_config = user_settings.pop(
            'estimator_config', default_estimator)
        super().__init__(default_settings, user_settings)
        self.model = None

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
        self._initialize_model(
            self.extract_settings['filepath'].split('/')[-1].split('.')[0])
        self.model.read(self.extract_settings)

    def transform(self) -> None:
        """
        The transform process from the model step ETL.
        """
        if self.model is None:
            model_type = self.estimator_config['module'].split('.')[0]
            self._initialize_model(model_type)
            self.model.build_model(
                self.estimator_config, self.dataset.normalizations)
        if ModelActions.fit in self.transform_settings:
            self._fit(self.transform_settings['fit'])
        if ModelActions.predict in self.transform_settings:
            self.predictions = self._predict(
                self.transform_settings['predict'])

    def load(self) -> None:
        """
        The load process from the model step ETL.
        """
        self.model.save(self.load_settings)

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

    def _initialize_model(self, model_type: str) -> None:
        """
        Initialize the specific type of model.

        Args:
            model_type (str): the kind of model to initialize.
        """
        if model_type == base_model.ModelType.sklearn:
            self.model = sklearn_model.SklearnModel(self.estimator_type)
        else:
            raise model_exceptions.ModelDoesNotExists(model_type)

    def run(self, metadata: Dict) -> Dict:
        """
        Run the model steps. Using the model created run the ETL functions for
        the specific model: extract, transform and load.

        Args:
            metadata (Dict): the objects output from each different previous
                steps.

        Returns:
            metadata (Dict): the previous objects updated with the ones from
                the current steps: ?.
        """
        # Feed the model with the objects
        self.dataset = metadata['dataset']
        if metadata.get('model', None) is not None:
            self.model = metadata['model']

        self.execute(metadata)

        metadata.update({'model': self.model})
        return metadata
