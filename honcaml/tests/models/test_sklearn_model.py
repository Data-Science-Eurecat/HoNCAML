import copy
import numpy as np
import shutil
import tempfile
import unittest
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils import validation

from honcaml.data import tabular, normalization
from honcaml.models import sklearn_model
from honcaml.tests import utils


class SklearnTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = tabular.TabularDataset()
        self.dataset._dataset = utils.mock_up_read_dataframe()
        self.dataset._features = ['col1', 'col2']
        self.dataset._target = 'target'

        # Classifier dataset
        self.classifier_dataset = tabular.TabularDataset()
        self.classifier_dataset._dataset = \
            utils.mock_up_read_classifier_dataframe()
        self.classifier_dataset._features = ['col1', 'col2']
        self.classifier_dataset._target = 'target'

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_build_model_without_normalization(self):
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual('regressor', sk_model.estimator_type)

        problem_type = 'classification'
        model_config = {'module': 'sklearn.ensemble.RandomForestClassifier',
                        'params': {}}
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual('classifier', sk_model.estimator_type)

    def test_build_model_with_normalization(self):
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        # Normalize only features
        features_to_normalize = ['col1', 'col2']
        features_norm_config = {
            'features': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'params': {},
                'columns': features_to_normalize
            }
        }
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization(copy.deepcopy(features_norm_config))
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual('regressor', sk_model.estimator_type)

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
        target_to_normalize = 'target'
        target_norm_config = {
            'target': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'params': {},
                'columns': target_to_normalize
            }
        }
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization(copy.deepcopy(target_norm_config))
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual('regressor', sk_model.estimator_type)
        self.assertEqual(len(sk_model.estimator.steps), 1)
        self.assertIsInstance(
            sk_model.estimator['estimator'], TransformedTargetRegressor)

        # Normalize both
        both_norm_config = {
            **features_norm_config, **target_norm_config}
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization(both_norm_config)
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual('regressor', sk_model.estimator_type)
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

        problem_type = 'classification'
        model_config = {'module': 'sklearn.ensemble.RandomForestClassifier',
                        'params': {}}
        # Normalize only features
        features_to_normalize = ['col1', 'col2']
        features_norm_config = {
            'features': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'params': {},
                'columns': features_to_normalize
            }
        }
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization(copy.deepcopy(features_norm_config))
        sk_model.build_model(model_config, norm)
        self.assertIsNotNone(sk_model._estimator)
        self.assertEqual('classifier', sk_model.estimator_type)

        self.assertEqual(
            len(sk_model.estimator['pre_process'].transformers), 1)
        name, scaler, columns = \
            sk_model.estimator['pre_process'].transformers[0]
        self.assertEqual(name, 'features_normalization')
        self.assertListEqual(columns, features_to_normalize)
        self.assertFalse(
            isinstance(sk_model.estimator['estimator'],
                       TransformedTargetRegressor))

    def test_fit(self):
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        self.assertIsNone(
            validation.check_is_fitted(sk_model.estimator))

        problem_type = 'classification'
        model_config = {'module': 'sklearn.ensemble.RandomForestClassifier',
                        'params': {}}
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.classifier_dataset.values
        sk_model.fit(x, y)
        self.assertIsNone(
            validation.check_is_fitted(sk_model.estimator))

    def test_predict(self):
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        predictions = sk_model.predict(x)
        self.assertIsInstance(predictions, np.ndarray)

        problem_type = 'classification'
        model_config = {'module': 'sklearn.ensemble.RandomForestClassifier',
                        'params': {}}
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.classifier_dataset.values
        sk_model.fit(x, y)
        predictions = sk_model.predict(x)
        self.assertIsInstance(predictions, np.ndarray)

    def test_evaluate(self):
        # Evaluate regression problem
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        metrics = ['mean_absolute_error', 'root_mean_squared_error']
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.dataset.values
        sk_model.fit(x, y)
        metrics = sk_model.evaluate(x, y, metrics)
        self.assertIsInstance(metrics, dict)

        # Evaluate classification problem
        problem_type = 'classification'
        model_config = {'module': 'sklearn.ensemble.RandomForestClassifier',
                        'params': {}}
        metrics = ['accuracy_score', 'f1_score']
        sk_model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        sk_model.build_model(model_config, norm)
        x, y = self.classifier_dataset.values
        sk_model.fit(x, y)
        metrics = sk_model.evaluate(x, y, metrics)
        self.assertIsInstance(metrics, dict)
