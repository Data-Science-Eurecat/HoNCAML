import os
import shutil
import tempfile
import unittest
from sklearn.utils import validation
from unittest.mock import patch

from honcaml.data import tabular, normalization
from honcaml.models import sklearn_model, general
from honcaml.steps import model, base
from honcaml.tests import utils
from honcaml.tools.startup import params


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.extract = base.StepPhase.extract
        self.transform = base.StepPhase.transform
        self.load = base.StepPhase.load
        self._global_params = {'problem_type': 'regression'}

        self.dataset = tabular.TabularDataset()
        self.dataset._dataset = utils.mock_up_read_dataframe()
        self.dataset._features = ['col1', 'col2']
        self.dataset._target = ['target1', 'target2']

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    # Test _merge_settings method
    def test_merge_default_settings_and_user_settings(self):
        # Empty user settings
        empty_user_settings = dict()
        step = model.ModelStep(params['pipeline_steps']['model'],
                               empty_user_settings, self._global_params,
                               params['step_rules']['model'])
        self.assertDictEqual({}, step._step_settings)

        # Empty user_settings phases
        empty_phases = {'extract': None, 'transform': None, 'load': None}
        step = model.ModelStep(params['pipeline_steps']['model'],
                               empty_phases, self._global_params,
                               params['step_rules']['model'])
        self.assertDictEqual(step._step_settings, {
            'extract': {
                'filepath': 'models/sklearn.regressor.20220819-122417.sav'
            },
            'transform': {},
            'load': {
                'path': 'data/models/'
            }
        })

        # Fill transform cross-validation settings
        transform_user_settings = {
            'transform': {'fit': {'cross_validation': {}}},
        }
        step = model.ModelStep(params['pipeline_steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'])
        self.assertDictEqual(step._step_settings, {
            'transform': {
                'fit': {
                    'cross_validation': {
                        'strategy': 'k_fold',
                        'n_splits': 10,
                        'shuffle': True,
                        'random_state': 90,
                    }
                }
            }
        })

        # Override settings
        override_user_settings = {
            'extract': {'filepath': 'sklearn.regressor.1234.sav'},
            'transform': {'fit': {'cross_validation': {
                'strategy': 'repeated_k_fold', 'n_splits': 20}}
            },
            'load': {'path': 'models'}
        }
        step = model.ModelStep(params['pipeline_steps']['model'],
                               override_user_settings, self._global_params,
                               params['step_rules']['model'])
        self.assertDictEqual(step._step_settings, {
            'extract': {
                'filepath': 'sklearn.regressor.1234.sav'
            },
            'transform': {
                'fit': {
                    'cross_validation': {
                        'strategy': 'repeated_k_fold',
                        'n_splits': 20,
                        'shuffle': True,
                        'random_state': 90,
                    }
                }
            },
            'load': {
                'path': 'models'
            }
        })

    def test_validate_step(self):
        pass
        # TODO: refactor test once validate step applies
        # override_user_settings = {
        #     'extract': {'filepath': None},
        #     'transform': {'fit': {'cross_validation': {
        #         'strategy': None}}
        #     },
        #     'load': {'path': None}
        # }
        # with self.assertRaises(step_exception.StepValidationError):
        #     step = model.ModelStep(params['pipeline_steps']['model'],
        #                            override_user_settings,
        #                            params['step_rules']['model'])

    @patch('joblib.load')
    def test_extract(self, read_model_mockup):
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyper_parameters': {}}
        read_model_mockup.return_value = utils.mock_up_read_model(
            'sklearn', 'regression', model_config).estimator

        user_settings = {
            'extract': {'filepath': 'sklearn.1234.sav'},
        }
        step = model.ModelStep(params['pipeline_steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'])
        step._extract(step._extract_settings)
        self.assertIsInstance(step._model, sklearn_model.SklearnModel)
        self.assertIsNotNone(step._model.estimator)

    def test_transform(self):
        # Fit
        user_settings = {
            'transform': {'fit': None},
        }
        step = model.ModelStep(params['pipeline_steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'])
        step._dataset = self.dataset
        step._transform(step._transform_settings)
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))

        # Fit and cross-validate
        user_settings = {
            'transform': {'fit': {'cross_validation': {'n_splits': 2}}},
        }
        step = model.ModelStep(params['pipeline_steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'])
        step._dataset = self.dataset
        step._transform(step._transform_settings)
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))
        self.assertIsNotNone(step._cv_results)

        # Predict (also fit to avoid not fitted predictor error)
        user_settings = {
            'transform': {'predict': {'path': self.test_dir}, 'fit': None},
        }
        step = model.ModelStep(params['pipeline_steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'])
        step._dataset = self.dataset
        step._transform(step._transform_settings)
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('predictions')
                            for f in files_in_test_dir))

    def test_load(self):
        # User settings
        user_settings = {
            'load': {'path': self.test_dir}
        }
        norm = normalization.Normalization({})
        step = model.ModelStep(params['pipeline_steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'])

        step._model = general.initialize_model('sklearn', 'regression')

        step._model.build_model({
            'module': 'sklearn.ensemble.RandomForestRegressor',
            'hyper_parameters': {}
        }, norm)
        step._load(step._load_settings)
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('sklearn')
                            for f in files_in_test_dir))

    def test_fit(self):
        # Only fit
        transform_user_settings = {
            'transform': {'fit': None},
        }
        norm = normalization.Normalization({})
        step = model.ModelStep(params['pipeline_steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'])
        step._dataset = self.dataset
        step._model = general.initialize_model('sklearn', 'regression')
        step._model.build_model(
            {'module': 'sklearn.ensemble.RandomForestRegressor',
             'hyper_parameters': {}}, norm)
        step._fit(step._transform_settings['fit'])
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))

        # Fit and cross-validation
        transform_user_settings = {
            'transform': {'fit': {'cross_validation': {'n_splits': 2}}},
        }
        norm = normalization.Normalization({})
        step = model.ModelStep(params['pipeline_steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'])
        step._dataset = self.dataset
        step._model = general.initialize_model('sklearn', 'regression')
        step._model.build_model(
            {'module': 'sklearn.ensemble.RandomForestRegressor',
             'hyper_parameters': {}}, norm)
        step._fit(step._transform_settings['fit'])
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))
        self.assertIsNotNone(step._cv_results)

    def test_predict(self):
        # Predict
        # TODO: mock save predictions
        transform_user_settings = {
            'transform': {'predict': {'path': self.test_dir}},
        }
        norm = normalization.Normalization({})
        step = model.ModelStep(params['pipeline_steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'])
        step._dataset = self.dataset
        step._model = general.initialize_model('sklearn', 'regression')
        step._model.build_model(
            {'module': 'sklearn.ensemble.RandomForestRegressor',
             'hyper_parameters': {}}, norm)
        step._fit({'fit': None})
        step._predict(step._transform_settings['predict'])
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('predictions')
                            for f in files_in_test_dir))

    def test_run(self):
        # TODO: make test
        pass
