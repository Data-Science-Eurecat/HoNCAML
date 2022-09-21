import unittest
from unittest.mock import patch
from sklearn.utils import validation
import numpy as np

from src.tests import utils
from src.tools.startup import params
from src.models import sklearn_model
from src.data import tabular


class SklearnTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = tabular.TabularDataset()
        self.dataset._dataset = utils.mock_up_read_dataframe()
        self.dataset._features = ['col1', 'col2']
        self.dataset._target = ['target1', 'target2']

    @patch('joblib.load')
    def test_read(self, read_model_mockup):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        read_model_mockup.return_value = utils.mock_up_read_model(
            'sklearn', estimator_type, model_config)._estimator

        sk_model = sklearn_model.SklearnModel(estimator_type)
        sk_model.read(params['pipeline_steps']['model']['extract'])
        self.assertIsNotNone(sk_model._estimator)

    def test_build_model(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        normalizations = {}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        sk_model.build_model(model_config, normalizations)
        self.assertIsNotNone(sk_model._estimator)

        # TODO: missing normalizations testing

    def test_fit(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        normalizations = {}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        sk_model.build_model(model_config, normalizations)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        self.assertIsNone(
            validation.check_is_fitted(sk_model.estimator))

    def test_predict(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        normalizations = {}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        sk_model.build_model(model_config, normalizations)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        predictions = sk_model.predict(x)
        self.assertIsInstance(predictions, np.ndarray)

    def test_evaluate(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        normalizations = {}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        sk_model.build_model(model_config, normalizations)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        metrics = sk_model.evaluate(x, y)
        self.assertIsInstance(metrics, dict)

    def test_save(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        normalizations = {}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        sk_model.build_model(model_config, normalizations)
        sk_model.save(params['pipeline_steps']['model']['load'])
        # TODO: finish test assertions
