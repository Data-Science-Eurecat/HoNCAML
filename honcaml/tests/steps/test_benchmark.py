import os
import pandas as pd
import tempfile
import unittest
from unittest.mock import patch

from honcaml import benchmark as benchmark_modules
from honcaml.data import extract
from honcaml.exceptions import benchmark as benchmark_exceptions
from honcaml.steps import base
from honcaml.steps import benchmark
from honcaml.tools.startup import params


class ResultGridMockUp:
    """
    Mock up class to simulate ray.tune.ResultGrid. This class returns the
    results of hyper parameter search.
    """

    @staticmethod
    def get_dataframe():
        data = {
            'root_mean_squared_error': [4, 3, 2, 1],
            'mean_squared_error': [1] * 4,
            'mean_absolute_percentage_error': [2] * 4,
            'median_absolute_error': [3] * 4,
            'r2_score': [4] * 4,
            'mean_absolute_error': [5] * 4,
            'config/metric': ['metric'] * 4,
            'config/model_module': ['fake.module.Class1',
                                    'fake.module.Class2',
                                    'fake.module.Class2',
                                    'fake.module.Class1'],
            'config/param_space/max_features': ['sqrt', 'log', 'log', 'sqrt'],
            'config/param_space/n_estimators': [40, 30, 20, 10],
            'config/cv_split': [object] * 4,
            'config/dataset': [object] * 4
        }
        return pd.DataFrame(data)


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.extract = base.StepPhase.extract
        self.transform = base.StepPhase.transform
        self.load = base.StepPhase.load
        self._global_params = {'problem_type': 'regression'}

        self.settings = {
            'transform': {
                'metrics': [
                    'mean_squared_error',
                    'mean_absolute_percentage_error',
                    'median_absolute_error',
                    'r2_score',
                    'mean_absolute_error',
                    'root_mean_squared_error',
                ],
                'models': {
                    'sklearn.ensemble.RandomForestRegressor': {
                        'n_estimators': {
                            'method': 'randint', 'values': [2, 110]},
                        'max_features': {
                            'method': 'choice', 'values': ['sqrt', 'log']}
                    },
                    'sklearn.linear_model.LinearRegression': {
                        'fit_intercept': {
                            'method': 'choice', 'values': [True, False]}
                    }
                },
                'cross_validation': {
                    'module': 'sklearn.model_selection.KFold',
                    'params': {
                        'n_splits': 2,
                        'shuffle': True,
                        'random_state': 90
                    }
                },
                'tuner': {
                    'search_algorithm': {
                        'module': 'ray.tune.search.optuna.OptunaSearch',
                        'params': None
                    },
                    'tune_config': {
                        'num_samples': 5,
                        'metric': 'root_mean_squared_error',
                        'mode': 'min',
                    },
                    'run_config': {
                        'stop': {
                            'training_iteration': 2
                        }
                    },
                    'scheduler': {
                        'module': 'ray.tune.schedulers.HyperBandScheduler',
                        'params': None
                    }

                }

            },
            'load': {'path': 'honcaml_reports'}
        }

        self.user_settings = {}
        self.step_rules = {}
        self.execution_id = 'fake id'

        self.tuner_settings = self.settings['transform']['tuner']

        data = {
            'root_mean_squared_error': [4, 3, 2, 1],
            'config/metric': ['metric'] * 4,
            'config/model_module': ['fake.module.Class1',
                                    'fake.module.Class2',
                                    'fake.module.Class2',
                                    'fake.module.Class1'],
            'config/param_space/max_features': ['sqrt', 'log', 'log', 'sqrt'],
            'config/param_space/n_estimators': [40, 30, 20, 10],
        }
        self.result_df = pd.DataFrame(data)

    def test_retrieve_benchmark_class_sklearn(self):
        name = 'sklearn.ensemble.RandomForestRegressor'
        returned = benchmark.BenchmarkStep._retrieve_benchmark_class(name)
        self.assertEqual(returned, benchmark_modules.sklearn.SklearnBenchmark)

    def test_retrieve_benchmark_class_torch(self):
        name = 'torch'
        returned = benchmark.BenchmarkStep._retrieve_benchmark_class(name)
        self.assertEqual(returned, benchmark_modules.torch.TorchBenchmark)

    @patch('honcaml.tools.utils.import_library')
    def test_class_methods(self, import_library_mock_up):
        import_library_mock_up.return_value = object

        ben = benchmark.BenchmarkStep(
            self.settings, self.user_settings, self._global_params,
            self.step_rules, self.execution_id)

        # Test _clean_search_space
        models = self.settings['transform']['models']
        for model_name in models:
            self._benchmark = ben._retrieve_benchmark_class(model_name)
            search_space = models[model_name]

            param_space = self._benchmark.clean_search_space(search_space)

            for _, tune_method in param_space.items():
                self.assertNotIsInstance(tune_method, str)

        # Test when it generates a exception
        search_space_with_nonexistent_methods = {
            'n_estimators': {
                'method': 'fake_randint', 'values': [2, 110]},
            'max_features': {
                'method': 'fake_choice', 'values': ['sqrt', 'log']}
        }
        with self.assertRaises(benchmark_exceptions.TuneMethodDoesNotExists):
            _ = self._benchmark.clean_search_space(
                search_space_with_nonexistent_methods)

        # _clean_scheduler
        scheduler = ben._clean_scheduler(self.tuner_settings)
        self.assertIsNotNone(scheduler)
        self.assertIsInstance(scheduler, object)

        tuner_settings_without_scheduler = {}
        scheduler = ben._clean_scheduler(tuner_settings_without_scheduler)
        self.assertIsNone(scheduler)

        # Test get_metric_and_mode
        ben._get_metric_and_mode(self.tuner_settings)
        self.assertIsNotNone(ben._metric)
        self.assertEqual(ben._metric,
                         self.tuner_settings['tune_config']['metric'])
        self.assertIsNotNone(ben._mode)
        self.assertEqual(ben._mode,
                         self.tuner_settings['tune_config']['mode'])

        # Test _clean_reported_metrics
        ben._clean_reported_metrics(self.settings['transform'])
        ben._reported_metrics.sort()
        self.settings['transform']['metrics'].sort()
        self.assertIsInstance(ben._reported_metrics, list)
        self.assertListEqual(
            ben._reported_metrics, self.settings['transform']['metrics'])

        no_metrics_in_settings_ = {'metrics': []}
        ben._clean_reported_metrics(no_metrics_in_settings_)
        self.assertIsInstance(ben._reported_metrics, list)
        self.assertEqual(len(ben._reported_metrics), 1)
        self.assertEqual(
            ben._reported_metrics[0],
            self.tuner_settings['tune_config']['metric'])

        # Test _clean_search_algorithm
        search_algorithm = ben._clean_search_algorithm(self.tuner_settings)
        self.assertIsNotNone(search_algorithm)
        self.assertIsInstance(search_algorithm, object)

        tuner_settings_without_search_algorithm = {}
        search_algorithm = ben._clean_search_algorithm(
            tuner_settings_without_search_algorithm)
        self.assertIsNone(search_algorithm)

        # Test clean_tune_config
        tune_config = ben._clean_tune_config(self.tuner_settings)
        self.assertDictEqual(tune_config, self.tuner_settings['tune_config'])

        tune_config = ben._clean_tune_config({})
        self.assertIsNone(tune_config)

        # clean_run_config
        run_config = ben._clean_run_config(self.tuner_settings)
        self.assertDictEqual(run_config, self.tuner_settings['run_config'])

        run_config = ben._clean_run_config({})
        self.assertIsNone(run_config)

        # Test _filter_results_dataframe
        results = {
            'root_mean_squared_error': [4, 3, 2, 1],
            'config/cv_split': [object, object, object, object],
            'config/dataset': [object, object, object, object],
        }
        results_df = pd.DataFrame(results)
        results_df = ben._filter_results_dataframe(results_df)
        columns_to_remove = [
            'config/cv_split',
            'config/dataset',
        ]
        for col in columns_to_remove:
            self.assertNotIn(col, results_df.columns)

        # Test _sort_results
        df = ben._sort_results(results_df)
        self.assertListEqual(
            df['root_mean_squared_error'].values.tolist(), [1, 2, 3, 4])

        # Set mode to 'max'
        ben._mode = 'max'
        df = ben._sort_results(results_df)
        self.assertListEqual(
            df['root_mean_squared_error'].values.tolist(), [4, 3, 2, 1])

        # Test get_best_result with 'max' mode
        ben._get_best_result(self.result_df, results_df.dtypes.to_dict())
        best_fake_model = 'fake.module.Class1'
        self.assertEqual(ben._best_model, best_fake_model)

        best_fake_hyper_parameters = {
            'max_features': 'sqrt',
            'n_estimators': 40,
        }
        self.assertDictEqual(
            ben._best_hyper_parameters, best_fake_hyper_parameters)

        # Test get_best_metrics
        results = {
            'root_mean_squared_error': [4, 1, 2, 2],
            'fake_metrics_1': [1, 2, 3, 4],
            'fake_metric_2': [100, 200, 300, 400]
        }
        metrics_df = pd.DataFrame(results)
        best_metric = ben._get_best_metrics(metrics_df)
        self.assertDictEqual({'root_mean_squared_error': 4}, best_metric)

        # Test get_best_model_and_params_dict
        best_config_dict = ben.get_best_model_and_hyperparams_dict()
        self.assertIn('module', best_config_dict)
        self.assertIn('params', best_config_dict)
        self.assertEqual(best_fake_model, best_config_dict['module'])
        self.assertDictEqual(
            best_fake_hyper_parameters, best_config_dict['params'])

    @patch('ray.tune.Tuner.fit')
    def test_transform(self, mock_up_tuner_fit):
        mock_up_tuner_fit.return_value = ResultGridMockUp()

        ben = benchmark.BenchmarkStep(
            self.settings, self.user_settings, self._global_params,
            self.step_rules, self.execution_id)

        with tempfile.TemporaryDirectory() as temp_dir:

            # Override folder to store results
            ben._step_settings = {'load': {'path': os.path.join(
                temp_dir, 'fake_results')}}
            ben._transform_settings = params[
                'steps']['benchmark']['transform']
            ben._transform(self.settings['transform'])

            # Check if exists a folder for module and results.csv
            dir_content = os.listdir(temp_dir)
            self.assertEqual(len(dir_content), 1)
            self.assertEqual(dir_content[0], 'fake_results')

            results_folder = os.listdir(
                os.path.join(temp_dir, 'fake_results', self.execution_id))
            self.assertIn('results.csv', results_folder)
            self.assertIn(
                'sklearn.ensemble.RandomForestRegressor', results_folder)
            self.assertIn(
                'sklearn.linear_model.LinearRegression', results_folder)

    def test_extract_not_implemented(self):
        ben = benchmark.BenchmarkStep(
            self.settings, self.user_settings, self._global_params,
            self.step_rules, self.execution_id)

        with self.assertRaises(NotImplementedError):
            ben._extract({})

    @patch('ray.tune.Tuner.fit')
    def test_run(self, mock_up_tuner_fit):
        mock_up_tuner_fit.return_value = ResultGridMockUp()

        ben = benchmark.BenchmarkStep(
            {}, self.settings, self._global_params, self.step_rules,
            self.execution_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Override folder to store results
            ben._step_settings['load']['path'] = os.path.join(
                temp_dir, 'fake_results')
            fake_metadata = {
                'dataset': object
            }
            metadata = ben.run(fake_metadata)
            expected_metadata = {
                'module': 'fake.module.Class1',
                'params': {
                    'max_features': 'sqrt', 'n_estimators': 10}
            }

            # Check metadata dict
            self.assertDictEqual(metadata['model_config'], expected_metadata)

            # Check best model and best hyper parameters
            best_fake_model = 'fake.module.Class1'
            self.assertEqual(ben._best_model, best_fake_model)

            best_fake_hyper_parameters = {
                'max_features': 'sqrt',
                'n_estimators': 10,
            }
            self.assertDictEqual(
                ben._best_hyper_parameters, best_fake_hyper_parameters)

            # Check if exists a folder for module and results.csv
            dir_content = os.listdir(temp_dir)
            self.assertEqual(len(dir_content), 1)
            self.assertEqual(dir_content[0], 'fake_results')

            results_folder = os.listdir(
                os.path.join(temp_dir, 'fake_results', self.execution_id))
            self.assertIn('results.csv', results_folder)
            self.assertIn(
                'sklearn.ensemble.RandomForestRegressor', results_folder)
            self.assertIn(
                'sklearn.linear_model.LinearRegression', results_folder)

    def test_load(self):
        fake_best_model = 'fake.model'
        fake_best_hyperparams = {
            'fake_param_1': 20,
            'fake_param_2': [1, 2, 3]
        }

        # Test with save_best_config_params True
        settings = {
            'save_best_config_params': True
        }
        user_settings = {}
        ben = benchmark.BenchmarkStep(
            settings, user_settings, self._global_params,
            self.step_rules, self.execution_id)

        ben._best_model = fake_best_model
        ben._best_hyper_parameters = fake_best_hyperparams

        with tempfile.TemporaryDirectory() as temp_dir:
            ben._store_results_folder = temp_dir
            ben._load(settings)
            expected_file_path = os.path.join(
                temp_dir, 'best_config_params.yaml')

            self.assertTrue(os.path.exists(expected_file_path))

            result_dict = extract.read_yaml(expected_file_path)
            expected_dict = {
                'module': fake_best_model,
                'params': fake_best_hyperparams,

            }
            self.assertDictEqual(result_dict, expected_dict)

        # Test with save_best_config_params False
        settings = {
            'save_best_config_params': False
        }
        ben = benchmark.BenchmarkStep(
            settings, user_settings, self._global_params,
            self.step_rules, self.execution_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            ben._store_results_folder = temp_dir
            ben._load(settings)
            expected_file_path = os.path.join(
                temp_dir, 'best_config_params.yaml')

            self.assertFalse(os.path.exists(expected_file_path))

        # Test without save_best_config_params param
        settings = {}
        ben = benchmark.BenchmarkStep(
            settings, user_settings, self._global_params,
            self.step_rules, self.execution_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            ben._store_results_folder = temp_dir
            ben._load(settings)
            expected_file_path = os.path.join(
                temp_dir, 'best_config_params.yaml')

            self.assertFalse(os.path.exists(expected_file_path))
