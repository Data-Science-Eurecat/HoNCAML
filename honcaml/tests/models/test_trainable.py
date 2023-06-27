import copy
import numpy as np
from ray import tune
import unittest

from honcaml.data import tabular, transform
from honcaml.models import trainable
from honcaml.tests import utils


class TrainableTest(unittest.TestCase):
    def setUp(self) -> None:
        dataset = tabular.TabularDataset()
        dataset._dataset = utils.mock_up_read_dataframe()
        dataset._features = ['col1', 'col2']
        dataset._target = ['target1', 'target2']
        cv_split = {
            'module': 'sklearn.model_selection.KFold',
            'params': {'n_splits': 3}
        }
        self.config = {
            'model_module': 'sklearn.ensemble.RandomForestRegressor',
            'dataset': dataset,
            'cv_split': transform.CrossValidationSplit(**cv_split),
            'param_space': {
                'n_estimators': tune.randint(2, 10),
                'max_features': tune.choice(['sqrt', 'log2'])
            },
            'metric': 'root_mean_squared_error',
            'problem_type': 'regression'
        }

    def test_setup(self) -> None:
        train_obj = trainable.EstimatorTrainer(self.config)
        train_obj.setup(self.config)
        self.assertIsInstance(train_obj, trainable.EstimatorTrainer)
        self.assertDictEqual(train_obj.config, self.config)

    def test_step(self) -> None:
        seeds = [1, 4]
        param_spaces = copy.deepcopy(self.config['param_space'])
        # Iterate over parameters and get a sample from them
        # The idea is to emulate what tune is doing
        cv_results = []
        for seed in seeds:
            for param in list(param_spaces):
                param_value = param_spaces[param].sample(random_state=seed)
                self.config['param_space'][param] = param_value
                self.config['param_space']['random_state'] = seed
            train_obj = trainable.EstimatorTrainer(self.config)
            train_obj.setup(self.config)
            seed_results = train_obj.step()
            for metric in list(seed_results):
                seed_results[metric] = np.round(seed_results[metric], 2)
            cv_results.append(seed_results)
        expected_results = [
            {'root_mean_squared_error': 11.33},
            {'root_mean_squared_error': 9.52}]
        self.assertListEqual(expected_results, cv_results)