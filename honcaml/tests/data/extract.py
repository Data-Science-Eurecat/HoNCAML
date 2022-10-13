import unittest
from unittest.mock import patch, mock_open
import pandas as pd

from honcaml.tests import utils
from honcaml.data import extract
from honcaml.exceptions import data as data_exceptions


class ExtractTest(unittest.TestCase):
    def setUp(self):
        pass

    @patch("builtins.open", mock_open(read_data=utils.mock_up_yaml()))
    def test_read_yaml(self):
        result = extract.read_yaml('some_file.yaml')
        self.assertIsInstance(result, dict)
        self.assertDictEqual(result, {
            'key1': {
                'nest_key1': 1,
                'nest_key2': 2,
            },
            'key2': 'value',
        })

    @patch('pandas.read_csv')
    @patch('pandas.read_excel')
    def test_read_dataframe(self, read_csv_mock_up, read_excel_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        read_excel_mock_up.return_value = utils.mock_up_read_dataframe()

        # Extension .csv
        settings = {'filepath': 'some_file.csv'}
        result = extract.read_dataframe(settings)
        self.assertIsInstance(result, pd.DataFrame)

        # Extension .xls
        settings = {'filepath': 'some_file.xls'}
        result = extract.read_dataframe(settings)
        self.assertIsInstance(result, pd.DataFrame)

        # Extension .xlsx
        settings = {'filepath': 'some_file.xlsx'}
        result = extract.read_dataframe(settings)
        self.assertIsInstance(result, pd.DataFrame)

        # Invalid extension
        settings = {'filepath': 'some_file.aaa'}
        with self.assertRaises(data_exceptions.FileExtensionException):
            _ = extract.read_dataframe(settings)

    @patch('joblib.load')
    def test_read_model(self, read_model_mockup):
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        read_model_mockup.return_value = utils.mock_up_read_model(
            'sklearn', problem_type, model_config)._estimator
        settings = {'filepath': 'some_file.sav'}
        result = extract.read_model(settings)
        self.assertIsNotNone(result)
