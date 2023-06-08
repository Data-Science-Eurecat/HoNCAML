import copy
import numpy as np
import pandas as pd
import sklearn
import unittest

from honcaml.tests import utils
from honcaml.data import transform
from honcaml.exceptions import data as data_exceptions


class TransformTest(unittest.TestCase):
    def setUp(self):
        self.cv_params = {
            'module': 'sklearn.model_selection.KFold',
            'params': {'n_splits': 3}}

    def test_process_data(self):
        # TODO: make test once logic implemented
        dataset = utils.mock_up_read_dataframe()
        settings = {}
        transform.process_data(dataset, settings)

    def test_get_train_test_dataset(self):
        # Pandas dataframe dataset
        dataset = utils.mock_up_read_dataframe()
        train_idx = np.array([0, 1])
        test_idx = np.array([2])
        x_train, x_test = transform.get_train_test_dataset(
            dataset, train_idx, test_idx)
        self.assertIsInstance(x_train, pd.DataFrame)
        self.assertIsInstance(x_test, pd.DataFrame)
        self.assertTrue(x_train.equals(dataset.iloc[[0, 1]]))
        self.assertTrue(x_test.equals(dataset.iloc[[2]]))

        # Numpy array dataset
        dataset = utils.mock_up_read_dataframe().values
        train_idx = np.array([0, 1])
        test_idx = np.array([2])
        x_train, x_test = transform.get_train_test_dataset(
            dataset, train_idx, test_idx)
        self.assertIsInstance(x_train, np.ndarray)
        self.assertIsInstance(x_test, np.ndarray)
        self.assertTrue(np.array_equal(x_train, dataset[[0, 1]]))
        self.assertTrue(np.array_equal(x_test, dataset[[2]]))

        # Bad type dataset
        dataset = 'aaa'
        train_idx = np.array([0, 1])
        test_idx = np.array([2])
        with self.assertRaises(ValueError):
            transform.get_train_test_dataset(dataset, train_idx, test_idx)

    def test_cross_validation_split_module_params(self):
        cv = transform.CrossValidationSplit(**self.cv_params)
        self.assertEqual(self.cv_params['module'], cv._module)
        self.assertEqual(self.cv_params['params'], cv._params)

    # Test class CrossValidationSplit
    def test_cross_validation_creates_instance_based_on_strategy(self):
        cv = transform.CrossValidationSplit(**self.cv_params)
        cv_object = cv._create_cross_validation_instance()
        self.assertIsInstance(cv_object, sklearn.model_selection._split.KFold)

        # Test with fake strategy
        fake_module = {'module': 'sklearn.model_selection.ModuleNotExist'}
        cv = transform.CrossValidationSplit(**fake_module)
        with self.assertRaises(data_exceptions.CVModuleDoesNotExist):
            cv._create_cross_validation_instance()

    def test_cross_validation_returns_the_same_type_as_input_without_y(self):
        data = list(range(0, 100))
        array = np.array(data)
        cv = transform.CrossValidationSplit(**self.cv_params)
        for i, x_train, x_test, y_train, y_test in cv.split(array):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, np.ndarray))
            self.assertTrue(isinstance(x_test, np.ndarray))

            self.assertEqual(y_train, None)
            self.assertEqual(y_test, None)

        series = pd.Series(data)
        for i, x_train, x_test, y_train, y_test in cv.split(series):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.Series))
            self.assertTrue(isinstance(x_test, pd.Series))

            self.assertEqual(y_train, None)
            self.assertEqual(y_test, None)

        dataframe_data = {
            'col1': data,
            'col2': data,
            'col3': data,
            'col4': data,
        }
        dataframe = pd.DataFrame(dataframe_data)
        for i, x_train, x_test, y_train, y_test in cv.split(dataframe):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertEqual(y_train, None)
            self.assertEqual(y_test, None)

        # Test pandas dataframe with string index
        dataframe['str_column'] = dataframe['col1'].astype(str)
        dataframe = dataframe.set_index('str_column')

        for i, x_train, x_test, y_train, y_test in cv.split(dataframe):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertEqual(y_train, None)
            self.assertEqual(y_test, None)

        # Python list gets an Exception
        with self.assertRaises(ValueError):
            for _ in cv.split(data):
                pass

    def test_cross_validation_returns_the_same_type_as_input_with_y(self):
        data = list(range(0, 100))
        array = np.array(data)

        series = pd.Series(data)

        dataframe_data = {
            'col1': data,
            'col2': data,
            'col3': data,
            'col4': data,
        }
        dataframe = pd.DataFrame(dataframe_data)

        # Numpy array
        # x: ndarray target: ndarray
        cv = transform.CrossValidationSplit(**self.cv_params)
        for i, x_train, x_test, y_train, y_test in cv.split(array, array):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, np.ndarray))
            self.assertTrue(isinstance(x_test, np.ndarray))

            self.assertTrue(isinstance(y_train, np.ndarray))
            self.assertTrue(isinstance(y_test, np.ndarray))

        # x: ndarray target: pd.Series
        for i, x_train, x_test, y_train, y_test in cv.split(array, series):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, np.ndarray))
            self.assertTrue(isinstance(x_test, np.ndarray))

            self.assertTrue(isinstance(y_train, pd.Series))
            self.assertTrue(isinstance(y_test, pd.Series))

        # x: ndarray target: pd.DataFrame
        for i, x_train, x_test, y_train, y_test in cv.split(array, dataframe):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, np.ndarray))
            self.assertTrue(isinstance(x_test, np.ndarray))

            self.assertTrue(isinstance(y_train, pd.DataFrame))
            self.assertTrue(isinstance(y_test, pd.DataFrame))

        # Series
        # x: series target: ndarray
        for i, x_train, x_test, y_train, y_test in cv.split(series, array):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.Series))
            self.assertTrue(isinstance(x_test, pd.Series))

            self.assertTrue(isinstance(y_train, np.ndarray))
            self.assertTrue(isinstance(y_test, np.ndarray))

        # x: series target: series
        for i, x_train, x_test, y_train, y_test in cv.split(series, series):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.Series))
            self.assertTrue(isinstance(x_test, pd.Series))

            self.assertTrue(isinstance(y_train, pd.Series))
            self.assertTrue(isinstance(y_test, pd.Series))

        # x: series target: DataFrame
        for i, x_train, x_test, y_train, y_test in cv.split(series, dataframe):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.Series))
            self.assertTrue(isinstance(x_test, pd.Series))

            self.assertTrue(isinstance(y_train, pd.DataFrame))
            self.assertTrue(isinstance(y_test, pd.DataFrame))

        # DataFrame
        # x: dataframe target: array
        for i, x_train, x_test, y_train, y_test in cv.split(dataframe, array):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertTrue(isinstance(y_train, np.ndarray))
            self.assertTrue(isinstance(y_test, np.ndarray))

        # x: dataframe target: series
        for i, x_train, x_test, y_train, y_test in cv.split(dataframe, series):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertTrue(isinstance(y_train, pd.Series))
            self.assertTrue(isinstance(y_test, pd.Series))

        # x: dataframe target: dataframe
        for i, x_train, x_test, y_train, y_test in \
                cv.split(dataframe, dataframe):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertTrue(isinstance(y_train, pd.DataFrame))
            self.assertTrue(isinstance(y_test, pd.DataFrame))

        # Test pandas dataframe with string index
        dataframe['str_column'] = dataframe['col1'].astype(str)
        dataframe = dataframe.set_index('str_column')

        # ndarray target
        for i, x_train, x_test, y_train, y_test in cv.split(dataframe, array):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertTrue(isinstance(y_train, np.ndarray))
            self.assertTrue(isinstance(y_test, np.ndarray))

        # series target
        for i, x_train, x_test, y_train, y_test in cv.split(dataframe, series):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertTrue(isinstance(y_train, pd.Series))
            self.assertTrue(isinstance(y_test, pd.Series))

        # dataframe target
        for i, x_train, x_test, y_train, y_test in \
                cv.split(dataframe, dataframe):
            self.assertTrue(isinstance(i, int))

            self.assertTrue(isinstance(x_train, pd.DataFrame))
            self.assertTrue(isinstance(x_test, pd.DataFrame))

            self.assertTrue(isinstance(y_train, pd.DataFrame))
            self.assertTrue(isinstance(y_test, pd.DataFrame))

        # Python list gets an Exception
        with self.assertRaises(ValueError):
            for _ in cv.split(data):
                pass

    def test_split_method_gets_split_number_from_1_to_n(self):
        split_cv_params = copy.deepcopy(self.cv_params)
        n_splits_list = [2, 5, 10]

        data = list(range(0, 100))
        array = np.array(data)
        for n_splits in n_splits_list:
            split_cv_params['params']['n_splits'] = n_splits
            cv = transform.CrossValidationSplit(**split_cv_params)
            iter_splits = []
            for i, _, _, _, _ in cv.split(array):
                self.assertTrue(isinstance(i, int))
                iter_splits.append(i)
            self.assertListEqual(iter_splits, list(range(1, n_splits + 1)))
