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
from honcaml.steps.model import ModelActions
from honcaml.tests import utils


class TabularTest(unittest.TestCase):
    def setUp(self):
        self.settings_with_csv = {
            'filepath': 'fake/file/path.csv',
            'features':
                ['col1', 'col2'],
            'target': 'target'
        }

        self.settings_with_excel = {
            'filepath': 'fake/file/path.xls',
            'features':
                ['col1', 'col2'],
            'target': 'target'
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
        self.assertIsInstance(target, str)
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

        # With target
        settings_one_target = copy.deepcopy(self.settings_with_csv)
        self.tabular_obj.read(settings_one_target)
        y = self.tabular_obj.y
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(y.shape), 1)

        # No target
        self.settings_with_csv.pop('target')
        self.tabular_obj.read(copy.deepcopy(self.settings_with_csv))
        with self.assertRaises(data_exception.TargetNotSet):
            self.tabular_obj.y

    @patch('pandas.read_csv')
    def test_values(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()

        # With target
        self.tabular_obj.read(self.settings_with_csv.copy())
        x, y = self.tabular_obj.values
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(np.array_equal(
            x, self.tabular_obj._dataset[['col1', 'col2']].values))
        self.assertTrue(np.array_equal(
            y, self.tabular_obj._dataset['target'].values))

        # No target
        self.settings_with_csv.pop('target')
        self.tabular_obj.read(self.settings_with_csv.copy())
        with self.assertRaises(data_exception.TargetNotSet):
            self.tabular_obj.values

    @patch('pandas.read_csv')
    def test_clean_dataset_for_model(self, read_csv_mock_up):
        read_csv_mock_up.return_value = utils.mock_up_read_dataframe()
        # Features and target correct
        settings = self.settings_with_csv.copy()
        self.tabular_obj._features = settings.pop('features')
        self.tabular_obj._target = settings.pop('target')
        dataset = extract.read_dataframe(settings)
        model_actions = [ModelActions.fit]
        cleaned_dataset = self.tabular_obj._clean_dataset_for_model(
            dataset, model_actions)
        self.assertIsInstance(cleaned_dataset, pd.DataFrame)
        self.assertTrue(cleaned_dataset.equals(
            dataset[self.tabular_obj._features + [self.tabular_obj._target]]))

        # Features not set
        settings = self.settings_with_csv.copy()
        settings.pop('features')
        self.tabular_obj._features = []
        self.tabular_obj._target = settings.pop('target')
        dataset = extract.read_dataframe(settings)
        model_actions = [ModelActions.fit]
        cleaned_dataset = self.tabular_obj._clean_dataset_for_model(
            dataset, model_actions)
        self.assertIsInstance(cleaned_dataset, pd.DataFrame)
        self.assertEqual(self.tabular_obj._features, ['col1', 'col2'])
        self.assertTrue(cleaned_dataset.equals(
            dataset[self.tabular_obj._features + [self.tabular_obj._target]]))

        # Features not set, target not exists
        settings = self.settings_with_csv.copy()
        settings.pop('features')
        self.tabular_obj._features = []
        settings.pop('target')
        self.tabular_obj._target = ['target_incorrect']
        dataset = extract.read_dataframe(settings)
        model_actions = [ModelActions.fit]
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            cleaned_dataset = self.tabular_obj._clean_dataset_for_model(
                dataset, model_actions)

        # Feature column does not exist
        settings = self.settings_with_csv.copy()
        settings.pop('features')
        self.tabular_obj._features = ['col_incorrect']
        self.tabular_obj._target = settings.pop('target')
        dataset = extract.read_dataframe(settings)
        model_actions = [ModelActions.fit]
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            cleaned_dataset = self.tabular_obj._clean_dataset_for_model(
                dataset, model_actions)

        # Target column does not exist
        settings = self.settings_with_csv.copy()
        self.tabular_obj._features = settings.pop('features')
        settings.pop('target')
        self.tabular_obj._target = ['target_incorrect']
        dataset = extract.read_dataframe(settings)
        model_actions = [ModelActions.fit]
        with self.assertRaises(data_exception.ColumnDoesNotExists):
            cleaned_dataset = self.tabular_obj._clean_dataset_for_model(
                dataset, model_actions)

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
        self.assertEqual(
            tabular_obj.target, self.settings_with_csv['target'])
        self.assertTrue(isinstance(tabular_obj.dataframe, pd.DataFrame))
        total_columns = self.settings_with_csv['features'] + [
            self.settings_with_csv['target']]
        for col in total_columns:
            self.assertIn(col, result_dataset)

        # Excel
        tabular_obj.read(self.settings_with_excel.copy())
        result_dataset = tabular_obj.dataframe
        self.assertListEqual(
            tabular_obj._features, self.settings_with_csv['features'])
        self.assertEqual(
            tabular_obj.target, self.settings_with_csv['target'])
        total_columns = self.settings_with_csv['features'] + [
            self.settings_with_csv['target']]
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
