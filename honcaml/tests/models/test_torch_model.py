import numpy as np
import os
import shutil
import tempfile
import torch
import unittest

from honcaml.data import tabular
from honcaml.models import torch_model
from honcaml.tests import utils
from honcaml.tools.startup import params


class TorchModelTest(unittest.TestCase):
    def setUp(self) -> None:
        # Default model configuration
        self.model_config = {
            'module': 'torch',
            'params': {
                'epochs': 3,
                'layers': [
                    {'module': 'torch.nn.Linear',
                     'params': {'out_features': 64}},
                    {'module': 'torch.nn.ReLU'},
                    {'module': 'torch.nn.Linear'}
                ],
                'loader': {'batch_size': 20, 'shuffle': True},
                'loss': {
                    'regression': {
                        'module': 'torch.nn.MSELoss'},
                    'classification': {
                        'module': 'torch.nn.BCEWithLogitsLoss'}
                },
                'optimizer': {
                    'module': 'torch.optim.SGD',
                    'params': {'lr': 0.001, 'momentum': 0.9}
                }
            }
        }
        # Regression dataset
        self.regression_dataset = tabular.TabularDataset()
        self.regression_dataset._dataset = utils.mock_up_read_dataframe()
        self.regression_dataset._features = ['col1', 'col2']
        self.regression_dataset._target = ['target1', 'target2']
        # Classifier dataset
        self.classification_dataset = tabular.TabularDataset()
        self.classification_dataset._dataset = \
            utils.mock_up_read_classifier_dataframe()
        self.classification_dataset._features = ['col1', 'col2']
        self.classification_dataset._target = ['target']
        # Test directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_import_estimator(self):
        problem_type = 'regression'
        self.model_config['params']['loss'] = self.model_config[
            'params']['loss'][problem_type]
        whole_input_dim = 10
        whole_output_dim = 1
        expected = torch.nn.Sequential(
            *[torch.nn.Linear(10, 64),
              torch.nn.ReLU(),
              torch.nn.Linear(64, 1)])
        result = torch_model.TorchModel._import_estimator(
            self.model_config['params']['layers'],
            whole_input_dim, whole_output_dim)
        self.assertTrue(str(expected), str(result))

    @unittest.mock.patch('joblib.load')
    def test_read(self, read_model_mockup):
        problem_type = 'regression'
        self.model_config['params']['loss'] = self.model_config[
            'params']['loss'][problem_type]
        read_model_mockup.return_value = utils.mock_up_read_model(
            'torch', problem_type, self.model_config, None,
            self.regression_dataset._features,
            self.regression_dataset._target)._estimator

        model = torch_model.TorchModel(problem_type)
        model.read(params['steps']['model']['extract'])
        self.assertIsNotNone(model._estimator)

    def test_build_model_regression(self):
        problem_type = 'regression'
        self.model_config['params']['loss'] = self.model_config[
            'params']['loss'][problem_type]
        model = torch_model.TorchModel(problem_type)
        model.build_model(
            self.model_config, None, self.regression_dataset._features,
            self.regression_dataset._target)
        self.assertIsNotNone(model._estimator)
        self.assertEqual('regressor', model.estimator_type)

    def test_build_model_classification(self):
        problem_type = 'classification'
        self.model_config['params']['loss'] = self.model_config[
            'params']['loss'][problem_type]
        model = torch_model.TorchModel(problem_type)
        model.build_model(
            self.model_config, None, self.classification_dataset._features,
            self.classification_dataset._target)
        self.assertIsNotNone(model._estimator)
        self.assertEqual('classifier', model.estimator_type)

    def test_fit_regression(self):
        problem_type = 'regression'
        self.model_config['params']['loss'] = self.model_config[
            'params']['loss'][problem_type]
        model = torch_model.TorchModel(problem_type)
        model.build_model(
            self.model_config, None, self.regression_dataset._features,
            self.regression_dataset._target)
        x, y = self.regression_dataset.values
        model.fit(x, y, **self.model_config['params'])
        expected = torch.nn.Sequential(
            *[torch.nn.Linear(10, 64),
              torch.nn.ReLU(),
              torch.nn.Linear(64, 1)])
        self.assertTrue(str(expected), str(model.estimator))
        self.regression_model = model

    def test_fit_classification(self):
        problem_type = 'classification'
        self.model_config['params']['loss'] = self.model_config[
            'params']['loss'][problem_type]
        model = torch_model.TorchModel(problem_type)
        model.build_model(
            self.model_config, None, self.classification_dataset._features,
            self.classification_dataset._target)
        x, y = self.classification_dataset.values
        model.fit(x, y, **self.model_config['params'])
        expected = torch.nn.Sequential(
            *[torch.nn.Linear(10, 64),
              torch.nn.ReLU(),
              torch.nn.Linear(64, 1)])
        self.assertTrue(str(expected), str(model.estimator))
        self.classification_model = model

    def test_predict_regression(self):
        self.test_fit_regression()
        x, y = self.regression_dataset.values
        predictions = self.regression_model.predict(
            x, self.model_config['params']['loader'])
        self.assertIsInstance(predictions, np.ndarray)

    def test_predict_classification(self):
        self.test_fit_classification()
        x, y = self.classification_dataset.values
        predictions = self.classification_model.predict(
            x, self.model_config['params']['loader'])
        self.assertIsInstance(predictions, np.ndarray)

    def test_evaluate_regression(self):
        self.test_fit_regression()
        metrics = ['mean_absolute_error', 'root_mean_squared_error']
        x, y = self.regression_dataset.values
        metrics = self.regression_model.evaluate(
            x, y, metrics, self.model_config['params']['loader'])
        self.assertIsInstance(metrics, dict)

    def test_evaluate_classification(self):
        self.test_fit_classification()
        metrics = ['accuracy_score', 'f1_score']
        x, y = self.classification_dataset.values
        metrics = self.classification_model.evaluate(
            x, y, metrics, self.model_config['params']['loader'])
        self.assertIsInstance(metrics, dict)

    def test_save(self):
        problem_type = 'regression'
        model = torch_model.TorchModel(problem_type)
        model.build_model(
            self.model_config, None, self.regression_dataset._features,
            self.regression_dataset._target)
        model.save({'path': self.test_dir})
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(
            any(f.startswith('torch') for f in files_in_test_dir))


if __name__ == '__main__':
    unittest.main()
