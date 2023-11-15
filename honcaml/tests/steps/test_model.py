import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from honcaml.data import normalization, tabular
from honcaml.models import general, sklearn_model
from honcaml.steps import base, model
from honcaml.tests import utils
from honcaml.tools.startup import params
from sklearn.utils import validation


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.extract = base.StepPhase.extract
        self.transform = base.StepPhase.transform
        self.load = base.StepPhase.load
        self._execution_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
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
        step = model.ModelStep(params['steps']['model'],
                               empty_user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        self.assertDictEqual({}, step._step_settings)

        # Fill transform cross-validation settings
        transform_user_settings = {
            'transform': {'fit': {'cross_validation': {}}},
        }
        step = model.ModelStep(params['steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)

        self.assertDictEqual(
            step._step_settings,
            {'transform': {
                'fit': params['steps']['model']['transform']['fit']
            }})

        # Override settings
        override_user_settings = {
            'extract': {'filepath': 'sklearn.regressor.1234.sav'},
            'transform': {'fit': {'cross_validation': {
                'module': 'sklearn.model_selection.RepeatedKFold',
                'params': {'n_splits': 20}}}
            },
            'load': {'path': 'models'}
        }
        step = model.ModelStep(params['steps']['model'],
                               override_user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        self.assertDictEqual(step._step_settings, {
            'extract': {
                'filepath': 'sklearn.regressor.1234.sav'
            },
            'transform': {
                'fit': {
                    'estimator': params['steps']['model']['transform'][
                        'fit']['estimator'],
                    'cross_validation': {
                        'module': 'sklearn.model_selection.RepeatedKFold',
                        'params': {'n_splits': 20}
                    },
                    'metrics': params['steps']['model']['transform'][
                        'fit']['metrics']
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
        #     step = model.ModelStep(params['steps']['model'],
        #                            override_user_settings,
        #                            params['step_rules']['model'])

    @patch('joblib.load')
    def test_extract(self, read_model_mockup):
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        read_model_mockup.return_value = utils.mock_up_read_model(
            'sklearn', 'regression', model_config).estimator

        user_settings = {
            'extract': {'filepath': 'sklearn.1234.sav'},
        }
        step = model.ModelStep(params['steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        step._extract(step._extract_settings)
        self.assertIsInstance(step._model, sklearn_model.SklearnModel)
        self.assertIsNotNone(step._model.estimator)

    def test_transform(self):
        # Fit
        user_settings = {
            'global': {'problem_type': 'regression'},
            'transform': {'fit': None},
            'load': {'path': 'data/models'},
        }
        step = model.ModelStep(params['steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        step._transform_settings['fit']['estimator'] = params[
            'steps']['model']['transform']['fit']['estimator'][
                self._global_params['problem_type']]
        step._transform_settings['fit']['metrics'] = params[
            'steps']['model']['transform']['fit']['metrics'][
                self._global_params['problem_type']]

        step._dataset = self.dataset
        step._transform(step._transform_settings)
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))

        # Fit and cross-validate
        user_settings = {
            'transform': {'fit': {'cross_validation': None}},
            'load': {'path': 'data/models'}
        }
        step = model.ModelStep(params['steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        step._transform_settings['fit']['estimator'] = params[
            'steps']['model']['transform']['fit']['estimator'][
                self._global_params['problem_type']]
        step._transform_settings['fit']['metrics'] = params[
            'steps']['model']['transform']['fit']['metrics'][
                self._global_params['problem_type']]

        # TODO: Mock save results
        step._dataset = self.dataset
        step._transform(step._transform_settings)
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))
        self.assertIsNotNone(step._cv_results)

        # Predict (also fit to avoid not fitted predictor error)
        user_settings = {
            'transform': {'predict': {'path': self.test_dir}, 'fit': None},
            'load': {'path': 'data/models'}
        }
        step = model.ModelStep(params['steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        step._transform_settings['fit']['estimator'] = params[
            'steps']['model']['transform']['fit']['estimator'][
                self._global_params['problem_type']]
        step._transform_settings['fit']['metrics'] = params[
            'steps']['model']['transform']['fit']['metrics'][
                self._global_params['problem_type']]
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
        step = model.ModelStep(params['steps']['model'],
                               user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)

        step._model = general.initialize_model('sklearn', 'regression')

        step._model.build_model({
            'module': 'sklearn.ensemble.RandomForestRegressor',
            'params': {}
        }, norm)
        step._load(step._load_settings)
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('sklearn')
                            for f in files_in_test_dir))

    def test_fit(self):
        # Only fit
        transform_user_settings = {
            'transform': {'fit': None},
            'load': {'path': 'data/models'}
        }
        norm = normalization.Normalization({})
        step = model.ModelStep(params['steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        step._transform_settings['fit']['estimator'] = params[
            'steps']['model']['transform']['fit']['estimator'][
                self._global_params['problem_type']]
        step._transform_settings['fit']['metrics'] = params[
            'steps']['model']['transform']['fit']['metrics'][
                self._global_params['problem_type']]
        step._dataset = self.dataset
        step._model = general.initialize_model('sklearn', 'regression')
        step._model.build_model(
            {'module': 'sklearn.ensemble.RandomForestRegressor',
             'params': {}}, norm)
        step._fit(step._transform_settings['fit'])
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))

        # Fit and cross-validation
        transform_user_settings = {
            'transform': {'fit': {'cross_validation': None}},
            'load': {'path': 'data/models'}
        }
        norm = normalization.Normalization({})
        step = model.ModelStep(params['steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        step._transform_settings['fit']['estimator'] = params[
            'steps']['model']['transform']['fit']['estimator'][
                self._global_params['problem_type']]
        step._transform_settings['fit']['metrics'] = params[
            'steps']['model']['transform']['fit']['metrics'][
                self._global_params['problem_type']]
        step._dataset = self.dataset
        step._model = general.initialize_model('sklearn', 'regression')
        step._model.build_model(
            {'module': 'sklearn.ensemble.RandomForestRegressor',
             'params': {}}, norm)
        step._fit(step._transform_settings['fit'])
        self.assertIsNone(
            validation.check_is_fitted(step._model.estimator))
        self.assertIsNotNone(step._cv_results)

    def test_generate_predictions_df(self):
        x = pd.DataFrame(
            columns=['f1', 'f2'],
            data=[[0, 'Str1'],
                  [1, 'Str2']])
        predictions = np.array([2, 5])
        target = 'target'
        expected = pd.DataFrame(
            columns=['f1', 'f2', 'target'],
            data=[[0, 'Str1', 2],
                  [1, 'Str2', 5]])
        result = model.ModelStep._generate_predictions_df(
            x, predictions, target)
        pd.testing.assert_frame_equal(expected, result)

    def test_predict(self):
        transform_user_settings = {
            'transform': {
                'fit': None,
                'predict': {'path': self.test_dir}},
            'load': {'path': 'data/models'}
        }
        norm = normalization.Normalization({})
        step = model.ModelStep(params['steps']['model'],
                               transform_user_settings, self._global_params,
                               params['step_rules']['model'],
                               self._execution_id)
        step._dataset = self.dataset
        step._model = general.initialize_model('sklearn', 'regression')
        step._model.build_model(
            {'module': 'sklearn.ensemble.RandomForestRegressor',
             'params': {}}, norm)
        step._transform_settings['fit']['estimator'] = params[
            'steps']['model']['transform']['fit']['estimator'][
                self._global_params['problem_type']]
        step._transform_settings['fit']['metrics'] = params[
            'steps']['model']['transform']['fit']['metrics'][
                self._global_params['problem_type']]
        step._fit(step.transform_settings['fit'])
        step._predict(step._transform_settings['predict'])
        files_in_test_dir = os.listdir(self.test_dir)
        self.assertTrue(any(f.startswith('predictions')
                            for f in files_in_test_dir))

    def test_run(self):
        # TODO: make test
        pass


if __name__ == '__main__':
    unittest.main()
