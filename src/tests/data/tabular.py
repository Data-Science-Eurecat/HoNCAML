import copy
import pandas as pd
import unittest
from unittest.mock import patch

from src.data import tabular
from src.exceptions import data as data_exception
from src.tests import utils


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
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        read_excel_mock_up.return_value = utils.mock_up_read_dataframe()

        tabular_obj = tabular.TabularDataset()

        # CSV
        tabular_obj.read(self.settings_with_csv.copy())
        result_dataset = tabular_obj.dataset

        self.assertListEqual(
            tabular_obj._features, self.settings_with_csv['features'])
        self.assertListEqual(
            tabular_obj.target, self.settings_with_csv['target'])
        self.assertTrue(isinstance(tabular_obj.dataframe, pd.DataFrame))

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

        # If features list is empty, it includes all features without target
        settings_without_features_list = {
            'filepath': 'fake/file/path.csv',
            'target':
                ['target1']
        }
        tabular_obj.read(copy.deepcopy(settings_without_features_list))
        fake_df = utils.mock_up_read_dataframe()
        df_columns_without_target = fake_df \
            .drop(columns=settings_without_features_list['target']) \
            .columns.to_list()
        self.assertListEqual(tabular_obj.features, df_columns_without_target)

        # If features and target list is empty, it includes all columns
        settings_without_features_list = {
            'filepath': 'fake/file/path.csv',
        }
        tabular_obj.read(copy.deepcopy(settings_without_features_list))
        self.assertListEqual(tabular_obj.features, fake_df.columns.to_list())
        self.assertListEqual(tabular_obj.target, [])

        # Excel
        tabular_obj.read(self.settings_with_excel.copy())
        result_dataset = tabular_obj.dataset

        self.assertListEqual(
            tabular_obj._features, self.settings_with_csv['features'])
        self.assertListEqual(
            tabular_obj.target, self.settings_with_csv['target'])

        total_columns = \
            self.settings_with_csv['features'] + \
            self.settings_with_csv['target']
        for col in total_columns:
            self.assertIn(col, result_dataset)
