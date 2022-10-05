import numpy as np
import os
import shutil
import tempfile
import unittest
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils import validation
from unittest.mock import patch
import copy
from honcaml.data import tabular, normalization
from honcaml.models import sklearn_model
from honcaml.tests import utils
from honcaml.tools.startup import params


class SklearnTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = tabular.TabularDataset()
        self.dataset._dataset = utils.mock_up_read_dataframe()
        self.dataset._features = ['col1', 'col2']
        self.dataset._target = ['target1', 'target2']

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

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

    def test_build_model_without_normalization(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual(estimator_type, sk_model.estimator_type)

    def test_build_model_with_normalization(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        # Normalize only features
        features_to_normalize = ['col1', 'col2']
        features_norm_config = {
            'features': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'module_params': {},
                'columns': features_to_normalize
            }
        }
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization(copy.deepcopy(features_norm_config))
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual(estimator_type, sk_model.estimator_type)

        self.assertEqual(
            len(sk_model.estimator['pre_process'].transformers), 1)
        name, scaler, columns = \
            sk_model.estimator['pre_process'].transformers[0]
        self.assertEqual(name, 'features_normalization')
        self.assertListEqual(columns, features_to_normalize)
        self.assertFalse(
            isinstance(sk_model.estimator['estimator'],
                       TransformedTargetRegressor))

        # Normalize only target
        target_to_normalize = ['target1', 'target2']
        target_norm_config = {
            'target': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'module_params': {},
                'columns': target_to_normalize
            }
        }
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization(copy.deepcopy(target_norm_config))
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual(estimator_type, sk_model.estimator_type)
        self.assertEqual(len(sk_model.estimator.steps), 1)
        self.assertIsInstance(
            sk_model.estimator['estimator'], TransformedTargetRegressor)

        # Normalize both
        both_norm_config = {
            **features_norm_config, **target_norm_config}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization(both_norm_config)
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual(estimator_type, sk_model.estimator_type)
        # Features
        self.assertEqual(
            len(sk_model.estimator['pre_process'].transformers), 1)
        name, scaler, columns = \
            sk_model.estimator['pre_process'].transformers[0]
        self.assertEqual(name, 'features_normalization')
        self.assertListEqual(columns, features_to_normalize)
        # Target
        self.assertIsInstance(
            sk_model.estimator['estimator'], TransformedTargetRegressor)

    def test_fit(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        self.assertIsNone(
            validation.check_is_fitted(sk_model.estimator))

    def test_predict(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        predictions = sk_model.predict(x)
        self.assertIsInstance(predictions, np.ndarray)

    def test_evaluate(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        metrics = sk_model.evaluate(x, y)
        self.assertIsInstance(metrics, dict)

    def test_save(self):
        estimator_type = 'regressor'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        sk_model = sklearn_model.SklearnModel(estimator_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        sk_model.save({'path': self.test_dir})
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('sklearn.regressor')
                            for f in files_in_test_dir))
