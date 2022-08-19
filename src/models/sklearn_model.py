from typing import Dict, List
from src.data import extract
from src.models import base, general
from src.tools import utils
from sklearn import compose, pipeline


class SklearnModel(base.BaseModel):
    """
    Model base class.

    Attributes:

    """

    def __init__(self, estimator_type: str) -> None:
        super().__init__(estimator_type)

    def read(self, settings: Dict) -> None:
        """
        ETL model read. This function must be implemented by child classes.
        name format: estimator_type.unique_id
        """
        self.model = extract.read_model(settings)

    def build_model(self, model_config, normalizations):
        pipeline_steps = []
        if features_norm := normalizations.get('features', None) is not None:
            ct_feature = compose.ColumnTransformer(
                transformers=[('feature_scaler', utils.import_library(
                    features_norm['module']), features_norm['columns'])],
                remainder='passthrough')
            pipeline_steps.append(('feature_scaler', ct_feature))

        estimator = utils.import_library(
            model_config['module'], model_config['hyperparameters'])
        pipeline_steps.append(('estimator', estimator))

        self.estimator = pipeline.Pipeline(pipeline_steps)

        if target_norm := normalizations.get('target', None) is not None:
            ct_target = compose.ColumnTransformer(
                transformers=[('target_scaler', utils.import_library(
                    target_norm['module']), target_norm['columns'])],
                remainder='passthrough')

            self.estimator = compose.TransformedTargetRegressor(
                regressor=self.estimator, transformer=ct_target)

    def fit(self, X, y, **kwargs) -> None:
        """
        ETL model fit. This function must be implemented by child classes.
        """
        self.estimator = self.estimator.fit(X, y)

    def predict(self, X, **kwargs) -> List:
        """
        ETL model predict. This function must be implemented by child classes.
        """
        return self.estimator.predict(X)

    def evaluate(self, X, y, **kwargs) -> Dict:
        """
        ETL model evaluate. This function must be implemented by child classes.
        """
        y_pred = self.estimator.predict(X)
        if self.estimator_type == 'regressor':
            metrics = general.compute_regression_metrics(y, y_pred)
        else:
            metrics = general.compute_classification_metrics(y, y_pred)
        return metrics

    def save(self, settings: Dict) -> None:
        """
        ETL model save. This function must be implemented by child classes.
        format: estimator_type.unique_id
        """
        filename = utils.generate_unique_id(
            self.estimator_type, adding_uuid=True)
        pass
