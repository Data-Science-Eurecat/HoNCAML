import copy
import numpy as np
import os
import pandas as pd
import shutil
import tempfile
import unittest
from unittest.mock import patch

from honcaml.data import tabular, extract
from honcaml.exceptions import data as data_exception
from honcaml.tests import utils


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

        self.tabular_obj = tabular.TabularDataset()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    @patch('pandas.read_csv')
    def test_features(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        self.tabular_obj.read(self.settings_with_csv.copy())
        features = self.tabular_obj.features
        self.assertIsInstance(features, list)
        self.assertEqual(features, self.tabular_obj._features)

    @patch('pandas.read_csv')
    def test_target(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        self.tabular_obj.read(self.settings_with_csv.copy())
        target = self.tabular_obj.target
        self.assertIsInstance(target, list)
        self.assertEqual(target, self.tabular_obj._target)

    @patch('pandas.read_csv')
    def test_dataframe(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        self.tabular_obj.read(self.settings_with_csv.copy())
        dataframe = self.tabular_obj.dataframe
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertTrue(dataframe.equals(self.tabular_obj._dataset))

    @patch('pandas.read_csv')
    def test_x(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        self.tabular_obj.read(self.settings_with_csv.copy())
        x = self.tabular_obj.x
        self.assertIsInstance(x, pd.DataFrame)
        self.assertListEqual(x.columns.to_list(), ['col1', 'col2'])

    @patch('pandas.read_csv')
    def test_y(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        # Only one target
        settings_one_target = copy.deepcopy(self.settings_with_csv)
        del settings_one_target['target'][1]
        self.tabular_obj.read(settings_one_target)
        y = self.tabular_obj.y
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(y.shape[1], 1)

        # Multiple target
        self.tabular_obj.read(copy.deepcopy(self.settings_with_csv))
        y = self.tabular_obj.y
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(np.array_equal(
            y, self.tabular_obj._dataset[['target1', 'target2']].values))

    @patch('pandas.read_csv')
    def test_values(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        self.tabular_obj.read(self.settings_with_csv.copy())
        x, y = self.tabular_obj.values
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(np.array_equal(
            x, self.tabular_obj._dataset[['col1', 'col2']].values))
        self.assertTrue(np.array_equal(
            y, self.tabular_obj._dataset[['target1', 'target2']].values))

    @patch('pandas.read_csv')
    def test_clean_dataset(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        # Features and target correct
        settings = self.settings_with_csv.copy()
        self.tabular_obj._features = settings.pop('features')
        self.tabular_obj._target = settings.pop('target')
        dataset = extract.read_dataframe(settings)
        cleaned_dataset = self.tabular_obj._clean_dataset(dataset)
        self.assertIsInstance(cleaned_dataset, pd.DataFrame)
        self.assertTrue(cleaned_dataset.equals(
            dataset[self.tabular_obj._features + self.tabular_obj._target]))

        # Features not set
        settings = self.settings_with_csv.copy()
        settings.pop('features')
        self.tabular_obj._features = []
        self.tabular_obj._target = settings.pop('target')
        dataset = extract.read_dataframe(settings)
        cleaned_dataset = self.tabular_obj._clean_dataset(dataset)
        self.assertIsInstance(cleaned_dataset, pd.DataFrame)
        self.assertEqual(self.tabular_obj._features, ['col1', 'col2'])
        self.assertTrue(cleaned_dataset.equals(
            dataset[self.tabular_obj._features + self.tabular_obj._target]))

        # Features not set, target not exists
        settings = self.settings_with_csv.copy()
        settings.pop('features')
        self.tabular_obj._features = []
        settings.pop('target')
        self.tabular_obj._target = ['target0']
        dataset = extract.read_dataframe(settings)
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            self.tabular_obj._clean_dataset(dataset)

        # Feature column does not exist
        settings = self.settings_with_csv.copy()
        settings.pop('features')
        self.tabular_obj._features = ['col0']
        self.tabular_obj._target = settings.pop('target')
        dataset = extract.read_dataframe(settings)
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            self.tabular_obj._clean_dataset(dataset)

        # Target column does not exist
        settings = self.settings_with_csv.copy()
        self.tabular_obj._features = settings.pop('features')
        settings.pop('target')
        self.tabular_obj._target = ['target0']
        dataset = extract.read_dataframe(settings)
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            self.tabular_obj._clean_dataset(dataset)

    # Test class tabular.TabularDataset method read
    @patch('pandas.read_csv')
    @patch('pandas.read_excel')
    def test_read_dataset(self, read_csv_mock_up, read_excel_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        read_excel_mock_up.return_value = utils.mock_up_read_dataframe()

        tabular_obj = tabular.TabularDataset()

        # CSV
        tabular_obj.read(self.settings_with_csv.copy())
        result_dataset = tabular_obj.dataframe

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
        result_dataset = tabular_obj.dataframe

        self.assertListEqual(
            tabular_obj._features, self.settings_with_csv['features'])
        self.assertListEqual(
            tabular_obj.target, self.settings_with_csv['target'])

        total_columns = \
            self.settings_with_csv['features'] + \
            self.settings_with_csv['target']
        for col in total_columns:
            self.assertIn(col, result_dataset)

    @patch('pandas.read_csv')
    def test_preprocess(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        self.tabular_obj.read(self.settings_with_csv.copy())
        filepath = os.path.join(self.test_dir, 'dataset.csv')
        settings = {'filepath': filepath}
        self.tabular_obj.save(settings)
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(os.stat(filepath).st_size > 0)
