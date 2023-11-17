from sklearn import compose, pipeline
from typing import Dict, List, Callable

from honcaml.data import normalization
from honcaml.models import base, evaluate
from honcaml.tools import custom_typing as ct
from honcaml.tools import utils
from honcaml.tools.startup import logger


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
        self._model_type = base.ModelType.sklearn
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
            model_config['module'], model_config['params'])

    def build_model(self, model_config: Dict,
                    normalizations: normalization.Normalization,
                    *args: Dict) -> None:
        """
        Creates the sklearn estimator. It builds a sklearn pipeline to handle
        the requested normalizations.

        Args:
            model_config: Model configuration, i.e. module and its
                hyperparameters.
            normalizations: Definition of normalizations that applies to
                the dataset during the model pipeline.
            **kwargs: Extra parameters.
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
        if normalizations is not None and normalizations.target:
            estimator = compose.TransformedTargetRegressor(
                regressor=estimator,
                transformer=normalizations.target_normalizer)

        pipeline_steps.append(('estimator', estimator))
        self._estimator = pipeline.Pipeline(pipeline_steps)
        logger.debug(f'Model pipeline {self._estimator}')

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

    def evaluate(self, x: ct.Dataset, y: ct.Dataset, metrics: List,
                 **kwargs) -> Dict:
        """
        Evaluates the estimator on the given dataset.

        Args:
            x: Dataset features.
            y: Dataset target.
            metrics: Metrics to be computed.
            **kwargs: Extra parameters.

        Returns:
            Resulting metrics from the evaluation.
        """
        y_pred = self._estimator.predict(x)
        metrics = utils.ensure_input_list(metrics)
        metrics = evaluate.compute_metrics(y, y_pred, metrics)
        return metrics
