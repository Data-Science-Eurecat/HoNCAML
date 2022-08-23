import pandas as pd
import unittest
from unittest.mock import patch

from src.data import tabular
from src.exceptions import data as data_exception


def _mock_up_read_dataframe():
    data = {
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'target1': [10, 20, 30],
        'target2': [40, 50, 60]
    }
    return pd.DataFrame(data)


class TabularTest(unittest.TestCase):
    def setUp(self):
        self.settings_with_csv = {
            'filepath': 'fake/file/path.csv',
            'features':
                ['col1', 'col2'],
            'target':
                ['target1', 'target2']
        }

        self.settings_with_excel = {
            'filepath': 'fake/file/path.xls',
            'features':
                ['col1', 'col2'],
            'target':
                ['target1', 'target2']
        }

    # Test class tabular.TabularDataset method read
    @patch('pandas.read_csv')
    @patch('pandas.read_excel')
    def test_read_dataset(self, read_csv_mock_up, read_excel_mock_up):
        read_csv_mock_up.return_value = _mock_up_read_dataframe()
        read_excel_mock_up.return_value = _mock_up_read_dataframe()

        tabular_obj = tabular.TabularDataset()

        # CSV
        tabular_obj.read(self.settings_with_csv.copy())
        result_dataset = tabular_obj.dataset

        self.assertListEqual(
            tabular_obj.features, self.settings_with_csv['features'])
        self.assertListEqual(
            tabular_obj.target, self.settings_with_csv['target'])

        total_columns = \
            self.settings_with_csv['features'] + \
            self.settings_with_csv['target']
        for col in total_columns:
            self.assertIn(col, result_dataset)

        # Raise exception when column does not exist
        tabular_obj.read(self.settings_with_csv.copy())
        settings_with_wrong_features_columns = {
            'filepath': 'fake/file/path.csv',
            'features':
                ['col1', 'col22'],
            'target':
                ['target1', 'target2']
        }
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            tabular_obj.read(settings_with_wrong_features_columns)

        # Raise exception when column does not exist
        tabular_obj.read(self.settings_with_csv.copy())
        settings_with_wrong_target_columns = {
            'filepath': 'fake/file/path.csv',
            'features':
                ['col1', 'col2'],
            'target':
                ['target1', 'target22']
        }
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            tabular_obj.read(settings_with_wrong_target_columns)

        # Excel
        tabular_obj.read(self.settings_with_excel.copy())
        result_dataset = tabular_obj.dataset

        self.assertListEqual(
            tabular_obj.features, self.settings_with_csv['features'])
        self.assertListEqual(
            tabular_obj.target, self.settings_with_csv['target'])

        total_columns = \
            self.settings_with_csv['features'] + \
            self.settings_with_csv['target']
        for col in total_columns:
            self.assertIn(col, result_dataset)
