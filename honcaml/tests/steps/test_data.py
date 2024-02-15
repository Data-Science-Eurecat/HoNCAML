import copy
import datetime
import pandas as pd
import unittest
from unittest.mock import patch

from honcaml.steps import data, base
from honcaml.tests import utils
from honcaml.tools.startup import params


class DataTest(unittest.TestCase):
    def setUp(self):
        self.extract = base.StepPhase.extract
        self.transform = base.StepPhase.transform
        self.load = base.StepPhase.load
        self._execution_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self._global_params = {'problem_type': 'regression'}

    # Test _merge_settings method
    def test_merge_default_settings_and_user_settings(self):
        # Empty user settings
        empty_user_settings = dict()
        step = data.DataStep(params['steps']['data'],
                             empty_user_settings, self._global_params,
                             params['step_rules']['data'],
                             self._execution_id)
        self.assertDictEqual({}, step.step_settings)

        # user settings contains filepath
        user_settings = {
            'extract':
                {
                    'filepath': 'override_path/override_file.csv',
                    'new_param': 90
                },
        }
        step = data.DataStep(params['steps']['data'],
                             user_settings, self._global_params,
                             params['step_rules']['data'],
                             self._execution_id)
        # The result dict has the same number of phases
        self.assertEqual(1, len(step.step_settings))

        # Extract
        # Check the override params
        self.assertEqual(
            user_settings[self.extract]['filepath'],
            step.step_settings[self.extract]['filepath'])
        self.assertTrue(
            'new_param' in step.step_settings[self.extract])

        # Transform
        # Test: When user_settings does not have 'normalize' key it returns
        # default norm
        user_settings = {
            'transform': {
                'override_param': {
                    'param1': 1,
                    'param2': 2,
                }
            }
        }
        _ = data.DataStep(params['steps']['data'],
                          user_settings, self._global_params,
                          params['step_rules']['data'],
                          self._execution_id)
        self.assertDictEqual(
            user_settings['transform'],
            {'override_param': {'param1': 1, 'param2': 2},
             "encoding": {
                 "OHE": True,
                 "max_values": 100
             },
             'normalize': {
                 'features': {'module':
                              'sklearn.preprocessing.StandardScaler'},
                 'target': {
                     'module': 'sklearn.preprocessing.StandardScaler'}}})

        user_settings = {
            'transform': {
                'new_override_param': [1, 2, 3],
                'normalize': {
                    'features': {
                        'module': 'override_scaler',
                        'params': {'param1': 10},
                        'columns': ['col1', 'col2', 'col3']
                    },
                    'target': {
                        'params': {'param1': 10, 'with_std': False},
                        'columns': ['target']
                    }
                }
            }
        }
        # Check the default params for transform phase
        step = data.DataStep(params['steps']['data'],
                             user_settings, self._global_params,
                             params['step_rules']['data'],
                             self._execution_id)
        step_settings = step.step_settings
        user_transform = user_settings['transform']
        self.assertEqual(
            user_transform['normalize']['features']['module'],
            step_settings[self.transform]['normalize']['features']['module'])

        self.assertDictEqual(
            step_settings[self.transform]['normalize']['features'][
                'params'],
            user_transform['normalize']['features']['params'],
        )

        self.assertDictEqual(
            user_transform['normalize']['target']['params'],
            step_settings[self.transform]['normalize']['target'][
                'params'])

        self.assertListEqual(
            user_transform['normalize']['target']['columns'],
            step_settings[self.transform]['normalize']['target']['columns'])

        self.assertTrue('new_override_param' in step_settings[self.transform])

    # Test _extract method
    @patch('pandas.read_csv')
    @patch('pandas.read_excel')
    def test_extract_phase(self, read_csv_mock_up, read_excel_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        read_excel_mock_up.return_value = utils.mock_up_read_dataframe()

        empty_user_settings = {'extract': {'target': 'target'}}

        # When settings does not have features, it includes all features
        # without target.
        step = data.DataStep(params['steps']['data'],
                             empty_user_settings, self._global_params,
                             params['step_rules']['data'],
                             self._execution_id)
        step._extract(copy.deepcopy(step.extract_settings))

        self.assertEqual(step.dataset.target, step.extract_settings['target'])
        self.assertListEqual([], step.dataset.features)
        self.assertTrue(isinstance(step.dataset.dataframe, pd.DataFrame))

    def test_when_transform_not_contains_normalization_is_none(self):
        default_without_normalize_settings = {
            'data':
                {'transform': {
                    'param1': [1, 2, 3],
                    'param2': {'col1': 1}
                }}}
        empty_user_settings = {}
        step = data.DataStep(default_without_normalize_settings,
                             empty_user_settings, self._global_params,
                             params['step_rules']['data'],
                             self._execution_id)
        step._transform(step.step_settings)
        self.assertTrue(step.dataset.normalization is None)
