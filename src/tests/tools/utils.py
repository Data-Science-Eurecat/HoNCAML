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

    # Test method merge_settings
    def test_foo(self):
        pass
