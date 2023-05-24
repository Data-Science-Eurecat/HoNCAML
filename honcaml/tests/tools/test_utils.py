import argparse
from honcaml.tools import utils
import unittest
from sklearn.ensemble import RandomForestRegressor
from cerberus import Validator


class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    # Test import_library
    def test_import_library(self):
        # Import sklearn model without params
        module = 'sklearn.ensemble.RandomForestRegressor'
        result = utils.import_library(module)
        self.assertIsInstance(result, RandomForestRegressor)

        # Import sklearn model with params
        module = 'sklearn.ensemble.RandomForestRegressor'
        params = {'n_estimators': 10}
        result = utils.import_library(module, params)
        self.assertIsInstance(result, RandomForestRegressor)
        self.assertEqual(result.get_params()[
                         'n_estimators'], params['n_estimators'])

    # Test ensure_input_list method
    def test_ensure_input_list(self):
        # No list
        obj = 1
        expected = [1]
        result = utils.ensure_input_list(obj)
        self.assertListEqual(expected, result)

        # List
        obj = [1]
        expected = [1]
        result = utils.ensure_input_list(obj)
        self.assertListEqual(expected, result)

        # None object
        obj = None
        expected = []
        result = utils.ensure_input_list(obj)
        self.assertListEqual(expected, result)

    # Test generate_unique_id
    def test_generate_unique_id(self):
        estimator_module = None
        adding_uuid = False
        unique_id = utils.generate_unique_id(
            estimator_module, adding_uuid)
        self.assertRegex(unique_id, r'^\d{4}\d{2}\d{2}-\d{2}\d{2}\d{2}$')

        estimator_module = 'module'
        adding_uuid = False
        unique_id = utils.generate_unique_id(
            estimator_module, adding_uuid)
        self.assertRegex(
            unique_id, r'^module.\d{4}\d{2}\d{2}-\d{2}\d{2}\d{2}$')

        estimator_module = 'module'
        adding_uuid = False
        unique_id = utils.generate_unique_id(
            estimator_module, adding_uuid)
        self.assertRegex(
            unique_id, r'^module.\d{4}\d{2}\d{2}-\d{2}\d{2}\d{2}$')

        estimator_module = 'module'
        adding_uuid = True
        unique_id = utils.generate_unique_id(
            estimator_module, adding_uuid)
        regex_1 = r'module.\d{4}\d{2}\d{2}-\d{2}\d{2}\d{2}_'
        regex_2 = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-5][0-9a-f]{3}-'
        regex_3 = r'[089ab][0-9a-f]{3}-[0-9a-f]{12}'
        regex = r'^' + regex_1 + regex_2 + regex_3 + r'$'
        self.assertRegex(unique_id, regex)

    # Test update_dict_from_default_dict method
    def test_update_dict_from_default_dict(self):
        default_dict = {
            'key1': {
                'nested_key1': {
                    'nested_key2': 1
                }
            },
            'nested_key3': 2,
            'nested_key4': 3,
        }
        source_dict = None
        output_dict = utils.update_dict_from_default_dict(
            default_dict, source_dict)
        self.assertEqual(output_dict, {
            'key1': {
                'nested_key1': {
                    'nested_key2': 1
                }
            },
            'nested_key3': 2,
            'nested_key4': 3,
        })

        default_dict = {
            'test_steps': {
                'predict': {
                    'nested_key1.2': 1
                },
                'fit': True
            },
            'nested_key3': 2,
            'nested_key4': 3,
        }
        source_dict = {
            'test_steps': {
                'fit': True
            },
            'nested_key3': 2,
            'nested_key4': 3,
        }
        output_dict = utils.update_dict_from_default_dict(
            default_dict, source_dict)
        self.assertEqual(output_dict, {
            'test_steps': {
                'fit': True
            },
            'nested_key3': 2,
            'nested_key4': 3,
        })

        default_dict = {
            'pipeline_steps': {
                'nested_key1': {
                    'nested_key1.2': 1
                },
                'nested_key2': True
            },
            'nested_key3': 2,
            'nested_key4': 3,
        }
        source_dict = {
            'pipeline_steps': {
                'nested_key2': True
            },
            'nested_key3': 2,
            'nested_key4': 3,
        }
        output_dict = utils.update_dict_from_default_dict(
            default_dict, source_dict)
        self.assertEqual(output_dict, {
            'pipeline_steps': {
                'nested_key2': True
            },
            'nested_key3': 2,
            'nested_key4': 3,
        })

        source_dict = {}
        output_dict = utils.update_dict_from_default_dict(
            default_dict, source_dict)
        self.assertEqual(output_dict, {
            'pipeline_steps': {
                'nested_key1': {
                    'nested_key1.2': 1
                },
                'nested_key2': True
            },
            'nested_key3': 2,
            'nested_key4': 3,
        })

        source_dict = {
            'nested_key3': 20,
            'nested_key4': 30,
        }
        output_dict = utils.update_dict_from_default_dict(
            default_dict, source_dict)
        self.assertEqual(output_dict, {
            'pipeline_steps': {
                'nested_key1': {
                    'nested_key1.2': 1
                },
                'nested_key2': True
            },
            'nested_key3': 20,
            'nested_key4': 30,
        })

    # Test build_validator
    def test_build_validator(self):
        rules = {
            'rule1': {
                'nested_rule': [
                    {'required': True}
                ]
            }
        }
        validator = utils.build_validator(rules)
        self.assertIsInstance(validator, Validator)

        rules = {
            'rule1': [
                {'required': True}
            ]
        }
        validator = utils.build_validator(rules)
        self.assertIsInstance(validator, Validator)

    # Test build_validator_schema
    def test_build_validator_schema(self):
        rules = {
            'rule1': {
                'nested_rule1': [
                    {'required': True}
                ]
            }
        }
        schema = utils.build_validator_schema(rules)
        self.assertDictEqual(schema, {
            'type': 'dict',
            'keysrules': {
                'allowed': ['rule1']
            },
            'valuesrules': {
                'type': 'dict',
                'schema': {
                    'nested_rule1': {
                        'required': True
                    }
                },
            }
        })

    # Test build_validator_schema
    def test_get_config_generation_argname_value(self):
        args = argparse.Namespace(generate_advanced_config='path/file.yaml')
        expected = 'advanced', 'path/file.yaml'
        result = utils.get_config_generation_argname_value(args)
        self.assertTupleEqual(result, expected)
