from typing import Dict, List
from src.tools import custom_typing as ct
from src.data import extract
from src.models import base, general
from src.tools import utils
from sklearn import compose, pipeline


class SklearnModel(base.BaseModel):
    """
    Scikit Learn model wrapper.

    Attributes:
        estimator_type (str): the kind of estimator to be used. Valid values
            are `regressor` and `classifier`.
        estimator (TODO sklearn.base.BaseEstimator?): an sklearn estimator.
    """

    def __init__(self, estimator_type: str) -> None:
        """
        The class constructor which initializes the base class.

        Args:
            estimator_type (str): the kind of estimator to be used. Valid values
                are `regressor` and `classifier`.
        """
        super().__init__(estimator_type)
        self.default_estimator = 'sklearn.ensemble.RandomForestClassifier'

    def read(self, settings: Dict) -> None:
        """
        Read an estimator from disk.

        Args:
            settings (Dict): the parameter settings defining the read 
                operation.
        """
        self.model = extract.read_model(settings)

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

        # TODO: Check if model config defined else use default estimator
        if model_config is None:
            model_config = {'module': self.default_estimator,
                            'hyperparameters': {}}
        estimator = utils.import_library(
            model_config['module'], model_config['hyperparameters'])
        pipeline_steps.append(('estimator', estimator))

        self.estimator = pipeline.Pipeline(pipeline_steps)

        if normalizations.get('target', None) is not None:
            target_norm = normalizations['target']
            ct_target = compose.ColumnTransformer(
                transformers=[('target_scaler', utils.import_library(
                    target_norm['module']), target_norm['columns'])],
                remainder='passthrough')

            self.estimator = compose.TransformedTargetRegressor(
                regressor=self.estimator, transformer=ct_target)

    def fit(self, X: ct.Dataset, y: ct.Dataset, **kwargs) -> None:
        """
        Train the estimator on the specified dataset.

        Args:
            X (ct.Dataset): the dataset features.
            y (ct.Dataset): the dataset target.
            **kwargs (Dict): extra parameters.
        """
        self.estimator = self.estimator.fit(X, y)

    def predict(self, X: ct.Dataset, **kwargs) -> List:
        """
        Use the estimator to make predictions on the given dataset features.

        Args:
            X (ct.Dataset): the dataset features.
            **kwargs (Dict): extra parameters.

        Returns:
            predictions (List): the resulting predictions from the estimator.
        """
        return self.estimator.predict(X)

    def evaluate(self, X: ct.Dataset, y: ct.Dataset, **kwargs) -> Dict:
        """
        Evaluate the estimator on the given dataset.

        Args:
            X (ct.Dataset): the dataset features.
            y (ct.Dataset): the dataset target.
            **kwargs (Dict): extra parameters.

        Returns:
            metrics (Dict): the resulting metrics from the evaluation.
        """
        y_pred = self.estimator.predict(X)
        if self.estimator_type == 'regressor':
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
        filename = utils.generate_unique_id(
            self.estimator_type, adding_uuid=True)
        pass
