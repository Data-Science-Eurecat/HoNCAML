import copy
import unittest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.data import normalization


class NormalizationTest(unittest.TestCase):
    def setUp(self):
        self.settings = {
            'features': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'module_params': {'param1': 1, 'param2': 2},
                'columns': ['col1', 'col2', 'col3']
            },
            'target': {
                'module': 'sklearn.preprocessing.MinMaxScaler',
                'module_params': {'param1': 10, 'param2': 20},
                'columns': ['target1', 'target2']
            }
        }

    # Test when create new Normalization instance
    def test_when_create_new_instance_get_params_from_settings_dict(self):
        # Features and target normalization
        norm = normalization.Normalization(copy.deepcopy(self.settings))
        # Features
        self.assertListEqual(
            norm.features, self.settings['features']['columns'])
        features_normalizer = self.settings['features'].copy()
        del features_normalizer['columns']
        self.assertDictEqual(norm._features_normalizer, features_normalizer)
        # Target
        self.assertListEqual(norm.target, self.settings['target']['columns'])
        target_normalizer = self.settings['target'].copy()
        del target_normalizer['columns']
        self.assertDictEqual(norm._target_normalizer, target_normalizer)

        # No normalizations
        norm = normalization.Normalization({})
        # Features
        self.assertListEqual(norm.features, [])
        self.assertDictEqual(norm._features_normalizer, {})
        # Target
        self.assertListEqual(norm.target, [])
        self.assertDictEqual(norm._target_normalizer, {})

        # Only features normalization
        only_feature_settings = {
            'features': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'module_params': {'param1': 1, 'param2': 2},
                'columns': ['col1', 'col2', 'col3']
            },
        }
        norm = normalization.Normalization(
            copy.deepcopy(only_feature_settings))
        # Features
        self.assertListEqual(
            norm.features, self.settings['features']['columns'])
        features_normalizer = self.settings['features'].copy()
        del features_normalizer['columns']
        self.assertDictEqual(norm._features_normalizer, features_normalizer)
        # Target
        self.assertListEqual(norm.target, [])
        self.assertDictEqual(norm._target_normalizer, {})

        # Only target normalization
        only_target_settings = {
            'target': {
                'module': 'sklearn.preprocessing.MinMaxScaler',
                'module_params': {'param1': 10, 'param2': 20},
                'columns': ['target1', 'target2']
            }
        }
        norm = normalization.Normalization(copy.deepcopy(only_target_settings))
        # Features
        self.assertListEqual(norm.features, [])
        self.assertDictEqual(norm._features_normalizer, {})
        # Target
        self.assertListEqual(
            norm.target, only_target_settings['target']['columns'])
        target_normalizer = only_target_settings['target'].copy()
        del target_normalizer['columns']
        self.assertDictEqual(norm._target_normalizer, target_normalizer)

        # Only features normalization without module params
        only_feature_without_params_settings = {
            'features': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'columns': ['col1', 'col2', 'col3']
            },
        }
        norm = normalization.Normalization(
            copy.deepcopy(only_feature_without_params_settings))
        # Features
        self.assertListEqual(
            norm.features,
            only_feature_without_params_settings['features']['columns'])
        features_normalizer = \
            only_feature_without_params_settings['features'].copy()
        del features_normalizer['columns']
        self.assertDictEqual(norm._features_normalizer, features_normalizer)
        self.assertTrue('module_params' not in norm._features_normalizer)

        # When
        real_scalers_settings = {
            'features': {
                'module': 'sklearn.preprocessing.StandardScaler',
                'module_params': {'with_std': True},
                'columns': ['col1', 'col2', 'col3']
            },
            'target': {
                'module': 'sklearn.preprocessing.MinMaxScaler',
                'module_params': {},
                'columns': ['target1', 'target2']
            }
        }
        norm = normalization.Normalization(real_scalers_settings)

        f_scaler = norm.features_normalizer
        self.assertIsInstance(f_scaler, StandardScaler)

        t_scaler = norm.target_normalizer
        self.assertIsInstance(t_scaler, MinMaxScaler)
