from typing import Dict

from honcaml.data import load
from honcaml.data import transform
from honcaml.models import base as base_model
from honcaml.models import general, evaluate
from honcaml.steps import base
from honcaml.tools.startup import logger, params


class ModelActions:
    fit = 'fit'
    predict = 'predict'


class ModelStep(base.BaseStep):
    """
    The model step class is a step of the main pipeline. It performs different
    tasks such as train, predict and evaluate a model. The extract and load
    functions allow the steps to save or restore a model.

    Attributes:
        _estimator_type (str): The kind of estimator to be used. Valid values
            are `regressor` and `classifier`.
        _estimator_config (Dict): Definition of the estimator, being the module
            and their hyperparameters.
        _model (base_model.BaseModel): Model from this library wrapping the
            specific estimator.
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
        self._estimator_type = user_settings.pop('estimator_type', None)
        self._estimator_config = user_settings.pop(
            'estimator_config', params['default_regressor_estimator'])
        self._model = None
        self._dataset = None

    @property
    def model(self) -> base_model.BaseModel:
        """
        Getter method for the '_model' attribute.

        Returns:
            '_model' current value.
        """
        return self._model

    def _extract(self, settings: Dict) -> None:
        """
        Extract process from the model step ETL.

        Args:
            settings: Settings defining the extract ETL process.
        """
        self._model = general.initialize_model(
            settings['filepath'].split('/')[-1].split('.')[0],
            settings['filepath'].split('/')[-1].split('.')[1], )
        self._model.read(settings)

    def _transform(self, settings: Dict) -> None:
        """
        Transform process from the model step ETL.

        Args:
            settings: Settings defining the transform ETL process.
        """
        if self._model is None:
            model_type = self._estimator_config['module'].split('.')[0]
            self._model = general.initialize_model(
                model_type, self._estimator_type)
            self._model.build_model(
                self._estimator_config, self._dataset.normalization)
        if ModelActions.fit in settings:
            self._fit(settings['fit'])
        if ModelActions.predict in settings:
            self._predict(settings['predict'])

    def _load(self, settings: Dict) -> None:
        """
        Load process from the model step ETL.

        Args:
            settings: Settings defining the load ETL process.
        """
        self._model.save(settings)

    def _fit(self, settings: Dict) -> None:
        """
        Performs a training step on the whole dataset and a cross-validation
        one if specified.

        Args:
            settings: Ttraining and cross-validation configuration.
        """
        x, y = self._dataset.x, self._dataset.y
        if settings.get('cross_validation', None) is not None:
            # Run the cross-validation
            cv_split = transform.CrossValidationSplit(
<< << << < HEAD: src/steps/model.py
                settings['cross_validation'].pop('strategy'),
                **settings.pop('cross_validation'))
            self._cv_results = general.cross_validate_model(
                self._model, x, y, cv_split, **settings, **settings)


== == == =
                settings['cross_validation'].pop('strategy'))

            results=[]
            for split, x_train, x_test, y_train, y_test in
                    cv_split.split(x, y, **settings.pop('cross_validation')):
                logger.info(f'Running split {split} ...')
                self._model.fit(x_train, y_train, **settings)

                results.append(self._model.evaluate(
                    x_test, y_test, **settings))
            # Group cv metrics
            self._cv_results=evaluate.aggregate_cv_results(results)
>> >>>> > develop: honcaml/steps/model.py
            logger.info(self._cv_results)

        # Train the model with whole data
        logger.info(f'Training model with all data ...')
        self._model.fit(x, y, **settings)

    def _predict(self, settings: Dict) -> None:
        """
        Performs predictions for the whole dataset by using the model.

        Args:
            settings: Predict configuration.
        """
        x=self._dataset.x
        predictions=self._model.predict(x, **settings)
        load.save_predictions(predictions, settings)

    def run(self, metadata: Dict) -> Dict:
        """
        Runs the model steps. Using the model created run the ETL functions for
        the specific model: extract, transform and load.

        Args:
            metadata: Accumulated pipeline metadata.

        Returns:
            metadata: Updated pipeline with the best estimator as a model.
        """
        # Feed the model with the objects
        self._dataset=metadata['dataset']
        if metadata.get('model', None) is not None:
            self._model=metadata['model']

        self.execute()

        metadata.update({'model': self._model})
        return metadata
