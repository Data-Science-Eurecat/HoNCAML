import numpy as np
import pandas as pd
import unittest
from sklearn import model_selection

from src.data.transform import CrossValidationSplit
from src.exceptions import data as data_exceptions


class TransformTest(unittest.TestCase):
    def setUp(self):
        self.k_fold = 'k_fold'
        self.repeated_k_fold = 'repeated_k_fold'
        self.shuffle_split = 'shuffle_split'
        self.leave_one_out = 'leave_one_out'

    # Test class CrossValidationSplit
    def test_cross_validation_creates_instance_based_on_strategy(self):
        strategies = [
            (self.k_fold, model_selection.KFold),
            (self.repeated_k_fold, model_selection.RepeatedKFold),
            (self.shuffle_split, model_selection.ShuffleSplit),
            (self.leave_one_out, model_selection.LeaveOneOut),
        ]
        for strategy, instance in strategies:
            cv = CrossValidationSplit(strategy)
            cv_object = cv._create_cross_validation_instance()
            self.assertTrue(isinstance(cv_object, instance))

        # Test with fake strategy
        fake_strategy = 'fake'
        cv = CrossValidationSplit(fake_strategy)
        with self.assertRaises(data_exceptions.CVStrategyDoesNotExist):
            cv._create_cross_validation_instance()

    def test_cross_validation_returns_the_same_type_as_input_without_y(self):
        strategy = self.k_fold

        data = list(range(0, 100))
        array = np.array(data)
        cv = CrossValidationSplit(strategy)
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
        strategy = self.k_fold

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
        cv = CrossValidationSplit(strategy)
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
        strategy = self.k_fold
        n_splits_list = [2, 5, 10]

        data = list(range(0, 100))
        array = np.array(data)
        cv = CrossValidationSplit(strategy)
        for n_splits in n_splits_list:
            params = {
                'n_splits': n_splits,
            }
            iter_splits = []
            for i, _, _, _, _ in cv.split(array, **params):
                self.assertTrue(isinstance(i, int))
                iter_splits.append(i)
            self.assertListEqual(iter_splits, list(range(1, n_splits + 1)))
