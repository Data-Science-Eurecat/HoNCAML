from sklearn import compose, pipeline
from typing import Dict, List, Callable

from honcaml.data import extract, load, normalization
from honcaml.models import base, evaluate
from honcaml.tools import custom_typing as ct
from honcaml.tools import utils
from honcaml.tools.startup import logger


class SklearnModel(base.BaseModel):
    """
    Scikit Learn model wrapper.
    """

    def __init__(self, estimator_type: str) -> None:
        """
        The class constructor which initializes the base class.

        Args:
            estimator_type (str): the kind of estimator to be used. Valid
                values are `regressor` and `classifier`.
        """
        super().__init__(estimator_type)
        self._estimator = None

    @property
    def estimator(self) -> ct.SklearnModelTyping:
        """
        This is a getter method. This function returns the '_estimator'
        attribute.

        Returns:
            (TODO): the sklearn estimator.
        """
        return self._estimator

    @property
    def estimator_type(self) -> str:
        """
        This is a getter method. This function returns the '_estimator_type'
        attribute.

        Returns:
            (str): the estimator_type
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
            settings (Dict): the parameter settings defining the read
                operation.
        """
        self._estimator = extract.read_model(settings)

    def build_model(self, model_config: Dict,
                    normalizations: normalization.Normalization) -> None:
        """
        Create the sklearn estimator. It builds an sklearn pipeline to handle
        the requested normalizations.

        Args:
            model_config (Dict): the model configuration: the module and their
                hyperparameters.
            normalizations (normalization.Normalization): the definition of
                normalizations applied to the dataset during the
                model pipeline.
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
        logger.info(f'Model pipeline {self._estimator}')

    def fit(self, x: ct.Dataset, y: ct.Dataset, **kwargs) -> None:
        """
        Train the estimator on the specified dataset.

        Args:
            x (ct.Dataset): the dataset features.
            y (ct.Dataset): the dataset target.
            **kwargs (Dict): extra parameters.
        """
        self._estimator = self._estimator.fit(x, y)

    def predict(self, x: ct.Dataset, **kwargs) -> List:
        """
        Use the estimator to make predictions on the given dataset features.

        Args:
            x (ct.Dataset): the dataset features.
            **kwargs (Dict): extra parameters.

        Returns:
            predictions (List): the resulting predictions from the estimator.
        """
        return self._estimator.predict(x)

    def evaluate(self, x: ct.Dataset, y: ct.Dataset, **kwargs) -> Dict:
        """
        Evaluate the estimator on the given dataset.

        Args:
            x (ct.Dataset): the dataset features.
            y (ct.Dataset): the dataset target.
            **kwargs (Dict): extra parameters.

        Returns:
            metrics (Dict): the resulting metrics from the evaluation.
        """
        y_pred = self._estimator.predict(x)
        if self._estimator_type == 'regressor':
            metrics = evaluate.compute_regression_metrics(y, y_pred)
        else:
            metrics = evaluate.compute_classification_metrics(y, y_pred)
        return metrics

    def save(self, settings: Dict) -> None:
        """
        Store the estimator to disk.

        Args:
            settings (Dict): the parameter settings defining the store
                operation.
        """
        settings['filename'] = utils.generate_unique_id(
            base.ModelType.sklearn, self._estimator_type) + '.sav'
        load.save_model(self._estimator, settings)
