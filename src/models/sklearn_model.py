from typing import Dict, List
from src.tools import custom_typing as ct
from src.data import extract, load
from src.models import base, general
from src.tools import utils
from sklearn import compose, pipeline


class SklearnModel(base.BaseModel):
    """
    Scikit Learn model wrapper.
    """

    def __init__(self, estimator_type: str) -> None:
        """
        The class constructor which initializes the base class.

        Args:
            estimator_type (str): the kind of estimator to be used. Valid values
                are `regressor` and `classifier`.
        """
        super().__init__(estimator_type)

    @property  # TODO: set the return type
    def estimator(self):
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
        return self._estimator

    def read(self, settings: Dict) -> None:
        """
        Read an estimator from disk.

        Args:
            settings (Dict): the parameter settings defining the read 
                operation.
        """
        self._estimator = extract.read_model(settings)

    def build_model(self, model_config: Dict, normalizations: Dict) -> None:
        """
        Create the sklearn estimator. It builds an sklearn pipeline to handle
        the requested normalizations.

        Args:
            model_config (Dict): the model configuration: the module and their
                hyperparameters.
            normalizations (Dict): the definition of normalizations applied to
                the dataset during the model pipeline.
        """
        pipeline_steps = []
        if normalizations.get('features', None) is not None:
            features_norm = normalizations['features']
            ct_feature = compose.ColumnTransformer(
                transformers=[('feature_scaler', utils.import_library(
                    features_norm['module']), features_norm['columns'])],
                remainder='passthrough')
            pipeline_steps.append(('feature_scaler', ct_feature))

        estimator = utils.import_library(
            model_config['module'], model_config['hyperparameters'])
        pipeline_steps.append(('estimator', estimator))

        self._estimator = pipeline.Pipeline(pipeline_steps)

        if normalizations.get('target', None) is not None:
            target_norm = normalizations['target']
            ct_target = compose.ColumnTransformer(
                transformers=[('target_scaler', utils.import_library(
                    target_norm['module']), target_norm['columns'])],
                remainder='passthrough')

            self._estimator = compose.TransformedTargetRegressor(
                regressor=self._estimator, transformer=ct_target)

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
            metrics = general.compute_regression_metrics(y, y_pred)
        else:
            metrics = general.compute_classification_metrics(y, y_pred)
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
