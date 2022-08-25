import unittest

from src.steps import data, base


class DataTest(unittest.TestCase):
    def setUp(self):
        self.default_settings = {
            'data': {
                'extract': {
                    'filepath': 'data/raw/dataset.csv',
                    'target': ['target']},
                'transform': {
                    'some_param': 'some_value',
                    'normalize': {
                        'features': {
                            'module': 'sklearn.preprocessing.StandardScaler',
                        },
                        'target': {
                            'module': 'sklearn.preprocessing.StandardScaler',
                            'module_params': {'default_param': False},
                            'columns': ['target']
                        }
                    }
                }
            }
        }
        self.data_step = base.StepType.data
        self.extract = base.StepPhase.extract
        self.transform = base.StepPhase.transform
        self.load = base.StepPhase.load

    # Test _merge_settings method
    def test_merge_default_settings_and_user_settings(self):
        # Empty user settings
        empty_user_settings = dict()
        step = data.DataStep(self.default_settings, empty_user_settings)
        self.assertDictEqual(
            self.default_settings.get(self.data_step), step.step_settings)

        # user settings contains filepath
        user_settings = {
            'data': {
                'extract':
                    {'filepath': 'override_path/override_file.csv',
                     'new_param': 90},
            }
        }
        step = data.DataStep(self.default_settings, user_settings)
        # The result dict has the same number of phases
        self.assertEqual(
            len(self.default_settings.get(self.data_step)),
            len(step.step_settings))

        # Extract
        # Check the override params
        self.assertEqual(
            user_settings.get(self.data_step)[self.extract]['filepath'],
            step.step_settings[self.extract]['filepath'])
        self.assertTrue(
            'new_param' in step.step_settings[self.extract])
        # Check that result dict contains the other default params
        self.assertTrue('target' in step.step_settings[self.extract])

        # Transform
        # Test: When user_settings does not have 'normalize' key it returns
        # default norm
        user_settings = {
            'data': {
                'transform': {
                    'override_param': {
                        'param1': 1,
                        'param2': 2,
                    }}
            }
        }
        step = data.DataStep(self.default_settings, user_settings)
        self.assertDictEqual(
            self.default_settings[self.data_step][self.transform]['normalize'],
            step.step_settings[self.transform]['normalize'])
        self.assertDictEqual(
            user_settings[self.data_step]['transform'],
            {'override_param': {'param1': 1, 'param2': 2}})

        user_settings = {
            'data': {
                'transform': {
                    'new_override_param': [1, 2, 3],
                    'normalize': {
                        'features': {
                            'module': 'override_scaler',
                            'module_params': {'param1': 10},
                            'columns': ['col1', 'col2', 'col3']
                        },
                        'target': {
                            'module_params': {'param1': 10, 'with_std': False},
                            'columns': ['target1', 'target2']
                        }
                    }}
            }
        }
        # Check the default params for transform phase
        step = data.DataStep(self.default_settings, user_settings)
        step_settings = step.step_settings
        default_transform = self.default_settings[self.data_step]['transform']
        user_transform = user_settings[self.data_step]['transform']
        self.assertEqual(
            user_transform['normalize']['features']['module'],
            step_settings[self.transform]['normalize']['features']['module'])

        self.assertDictEqual(
            step_settings[self.transform]['normalize']['features']['module_params'],
            user_transform['normalize']['features']['module_params'],
        )

        self.assertDictEqual(
            default_transform['normalize']['target']['module_params'],
            step_settings[self.transform]['normalize']['target']['module_params'])

        self.assertListEqual(
            user_transform['normalize']['target']['columns'],
            step_settings[self.transform]['normalize']['target']['columns'])

        self.assertTrue('some_param' in step_settings[self.transform])
        self.assertTrue('new_override_param' in step_settings[self.transform])

    # Test normalization