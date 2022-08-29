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
        _estimator_type (str): the kind of estimator to be used. Valid
            values are `regressor` and `classifier`.
        _estimator_config (Dict): the definition of the estimator: the module
            and their hyperparameters.
        _model (base_model.BaseModel): the model from this library wrapping the
            specific estimator.
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
        self._estimator_type = user_settings.pop('estimator_type')
        self._estimator_config = user_settings.pop(
            'estimator_config', default_estimator)
        self._model = None

    @property
    def model(self) -> base_model.BaseModel:
        """
        This is a getter method. This function returns the '_model'
        attribute.

        Returns:
            (base_model.BaseModel): model instance.
        """
        return self._model

    def _validate_step(self) -> None:
        """
        Validates the settings for the step ensuring that the step has the
        mandatory keys to run.

        - estimator_type required (unless extract)
        - si estimator_config -> estimator_config.module required
        - si extract -> filepath required + format module_name.type.(.*)
        - si transform -> fit or predict
        - si fit -> si cross-val -> strategy required
        - si load -> path required
        """

        pass

    def _initialize_model(self, model_type: str, estimator_type: str) -> None:
        """
        Initialize the specific type of model.

        Args:
            model_type (str): the kind of model to initialize.
            estimator_type (str): the kind of estimator to be used. Valid
                values are `regressor` and `classifier`.
        """
        if model_type == base_model.ModelType.sklearn:
            self._model = sklearn_model.SklearnModel(estimator_type)
        else:
            raise model_exceptions.ModelDoesNotExists(model_type)

    def _extract(self, settings: Dict) -> None:
        """
        The extract process from the model step ETL.

        Args:
            settings (Dict): the settings defining the extract ETL process.
        """
        self._initialize_model(
            settings['filepath'].split('/')[-1].split('.')[0],
            settings['filepath'].split('/')[-1].split('.')[1],)
        self._model.read(settings)

    def _transform(self, settings: Dict) -> None:
        """
        The transform process from the model step ETL.

        Args:
            settings (Dict): the settings defining the transform ETL process.
        """
        if self._model is None:
            model_type = self._estimator_config['module'].split('.')[0]
            self._initialize_model(model_type, self._estimator_type)
            # TODO: Refactor the normalizations
            self._model.build_model(
                self._estimator_config, self._dataset.normalizations)
        if ModelActions.fit in settings:
            self._fit(settings['fit'])
        if ModelActions.predict in settings:
            self._predictions = self._predict(settings['predict'])

    def _load(self, settings: Dict) -> None:
        """
        The load process from the model step ETL.

        Args:
            settings (Dict): the settings defining the load ETL process.
        """
        self._model.save(settings)

    def _fit(self, settings: Dict) -> None:
        """
        The training function for the model step. It performs a training step
        on the whole dataset and a cross-validation one if specified.

        Args:
            settings (Dict): the training and cross-validation configuration.
        """
        x, y = self._dataset.values
        if settings.get('cross_validation', None) is not None:
            # Run the cross-validation
            cv_split = transform.CrossValidationSplit(
                settings['cross_validation'].pop('strategy'))

            results = []
            for split, x_train, x_test, y_train, y_test in \
                    cv_split.split(x, y, settings.pop('cross_validation')):
                # Afegir normalizations
                self._model.fit(x_train, y_train, **settings)

                results.append(self.model.evaluate(x_test, y_test, **settings))
        # Group cv metrics
        self._cv_results = general.aggregate_cv_results(results)
        # Train the model with whole data
        self.model.fit(x, y, **settings)

    def _predict(self, settings: Dict) -> List:
        """
        The predict function for the model step. It performs predictions for
        the whole dataset by using the model.

        Args:
            settings (Dict): the predict configuration.

        Returns:
            predictions (List): the prediction for each sample.
        """
        x = self._dataset.x
        predictions = self._model.predict(x, **settings)
        return predictions

    def run(self, metadata: Dict) -> Dict:
        """
        Run the model steps. Using the model created run the ETL functions for
        the specific model: extract, transform and load.

        Args:
            metadata (Dict): the objects output from each different previous
                steps.

        Returns:
            metadata (Dict): the previous objects updated with the ones from
                the current steps: the best estimator as a model from this
                library.
        """
        # Feed the model with the objects
        self._dataset = metadata['dataset']
        if metadata.get('model', None) is not None:
            self._model = metadata['model']

        self.execute(metadata)

        metadata.update({'model': self._model})
        return metadata
