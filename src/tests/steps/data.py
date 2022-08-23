import unittest

from src.steps import data, base


class TransformTest(unittest.TestCase):
    def setUp(self):
        self.default_settings = {
            'data': {'extract': {
                'filepath': 'data/raw/dataset.csv',
                'target': ['target']}}
        }
        self.step_type = base.StepType.data

    # Test _merge_settings method
    def test_merge_default_settings_and_user_settings(self):
        # Empty user settings
        empty_user_settings = dict()
        step = data.DataStep(self.default_settings, empty_user_settings)
        self.assertDictEqual(
            self.default_settings.get(self.step_type), step.step_settings)

        # user settings contains filepath
        user_settings = {
            'data': {
                'extract':
                    {'filepath': 'override_path/override_file.csv'}}
        }
        step = data.DataStep(self.default_settings, user_settings)
        # self.default_settings.get(self.step_type)
        self.assertEqual(
            user_settings.get(self.step_type)['extract']['filepath'],
            step.step_settings['filepath'])
