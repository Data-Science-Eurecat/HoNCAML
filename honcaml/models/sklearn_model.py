from sklearn import compose, pipeline
from typing import Dict, List, Callable

from honcaml.data import extract, load, normalization
from honcaml.models import base, evaluate
from honcaml.tools import custom_typing as ct
from honcaml.tools import utils
from honcaml.tools.startup import logger
from honcaml.exceptions import model as model_exceptions


class SklearnModel(base.BaseModel):
    """
    Scikit Learn model wrapper.
    """

    def __init__(self, problem_type: str) -> None:
        """
        Class constructor which initializes the base class.

        Args:
            problem_type (str): The kind of problem to be addressed. Valid
                values are `regression` and `classification`.
        """
        super().__init__(problem_type)
        self._estimator = None

    @property
    def estimator(self) -> ct.SklearnModelTyping:
        """
        Getter method for the '_estimator' attribute.

        Returns:
            '_estimator' current value.
        """
        return self._estimator

    @property
    def estimator_type(self) -> str:
        """
        Getter method for the '_estimator_type' attribute.

        Returns:
            '_estimator_type' current value.
        """
        return self._estimator_type

    @staticmethod
    def _import_estimator(model_config: dict) -> Callable:
        """
        Given a dict with model configuration, this function import the model
        and, it creates a new instance with the hyperparameters.

        Args:
            model_config (dict): a dict with the module and hyperparameters to
                import.

        Returns:
            (Callable): an instance of model with specific hyperparameters.
        """
        return utils.import_library(
            model_config['module'], model_config['hyperparameters'])

    def read(self, settings: Dict) -> None:
        """
        Read an estimator from disk.

        Args:
            settings: Parameter settings defining the read operation.
        """
        self._estimator = extract.read_model(settings)

    def build_model(self, model_config: Dict,
                    normalizations: normalization.Normalization) -> None:
        """
        Creates the sklearn estimator. It builds a sklearn pipeline to handle
        the requested normalizations.

        Args:
            model_config: Model configuration, i.e. module and its
                hyperparameters.
            normalizations: Definition of normalizations that applies to
                the dataset during the model pipeline.
        """
        pipeline_steps = []
        # Preprocessing
        pre_process_transformations = []
        if normalizations is not None and normalizations.features:
            features_norm = ('features_normalization',
                             normalizations.features_normalizer,
                             normalizations.features)
            pre_process_transformations.append(features_norm)
        # Adding more transformations here

        if pre_process_transformations:
            pre_process = compose.ColumnTransformer(
                transformers=pre_process_transformations,
                remainder='passthrough')
            pipeline_steps.append(('pre_process', pre_process))

        # Model
        estimator = self._import_estimator(model_config)
        # estimator = RandomForestRegressor()
        if normalizations is not None and normalizations.target:
            estimator = compose.TransformedTargetRegressor(
                regressor=estimator,
                transformer=normalizations.target_normalizer)

        pipeline_steps.append(('estimator', estimator))
        self._estimator = pipeline.Pipeline(pipeline_steps)
        logger.info(f'Model pipeline {self._estimator}')

    def fit(self, x: ct.Dataset, y: ct.Dataset, **kwargs) -> None:
        """
        Trains the estimator on the specified dataset.
        Args:
            x: Dataset features.
            y: Dataset target.
            **kwargs: Extra parameters.
        """
        self._estimator = self._estimator.fit(x, y)

    def predict(self, x: ct.Dataset, **kwargs) -> List:
        """
        Uses the estimator to make predictions on the given dataset features.

        Args:
            x: Dataset features.
            **kwargs: Extra parameters.

        Returns:
            Resulting predictions from the estimator.
        """
        return self._estimator.predict(x)

    def evaluate(self, x: ct.Dataset, y: ct.Dataset, **kwargs) -> Dict:
        """
        Evaluates the estimator on the given dataset.

        Args:
            x: Dataset features.
            y: Dataset target.
            **kwargs: Extra parameters.

        Returns:
            Resulting metrics from the evaluation.
        """
        y_pred = self._estimator.predict(x)
        if self._estimator_type == base.EstimatorType.regressor:
            metrics = evaluate.compute_regression_metrics(y, y_pred)
        elif self._estimator_type == base.EstimatorType.classifier:
            metrics = evaluate.compute_classification_metrics(y, y_pred)
        else:
            raise model_exceptions.EstimatorTypeNotAllowed(
                self._estimator_type)
        return metrics

    def save(self, settings: Dict) -> None:
        """
        Stores the estimator to disk.

        Args:
            settings: Parameter settings defining the store operation.
        """
        settings['filename'] = utils.generate_unique_id(
            base.ModelType.sklearn) + '.sav'
        load.save_model(self._estimator, settings)
