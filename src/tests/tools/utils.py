from src.tools import utils
import unittest


class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    # Test ensure_input_list method
    def test_ensure_input_list_no_list(self):
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

    # Test update_dict_from_default_dict method
    def test_update_dict_from_default_dict(self):
        default_dict = {
            'filepath': 'default/path/file.csv',
            'target': ['default_target1', 'default_target2'],
            'param4': 2
        }

        # Empty input dict
        result = utils.update_dict_from_default_dict({}, default_dict)
        self.assertEqual(len(result), len(default_dict))
        self.assertDictEqual(result, default_dict)

        # Dict with two params
        input_dict = {
            'filepath': 'not/override/file.csv',
            'new_param': 90
        }
        result = utils.update_dict_from_default_dict(input_dict, default_dict)
        self.assertEqual(len(result), len({**input_dict, **default_dict}))
        self.assertEqual(result['filepath'], input_dict['filepath'])
        self.assertEqual(result['target'], default_dict['target'])
        self.assertEqual(result['param4'], default_dict['param4'])
        self.assertTrue('new_param' in result)

        # Empty default dict
        result = utils.update_dict_from_default_dict(input_dict, {})
        self.assertEqual(len(result), len(input_dict))
        self.assertDictEqual(result, input_dict)

        # Input dict with 2 parameters
        input_dict = {
            'filepath': 'not/override/file.csv',
            'new_param': 90,
            'target': ['new_target']
        }
        result = utils.update_dict_from_default_dict(input_dict, default_dict)
        self.assertEqual(len(result), len({**input_dict, **default_dict}))
        self.assertEqual(result['filepath'], input_dict['filepath'])
        self.assertEqual(result['target'], input_dict['target'])
        self.assertEqual(result['param4'], default_dict['param4'])
        self.assertTrue('new_param' in result)
