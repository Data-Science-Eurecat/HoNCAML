import unittest

import numpy as np
import pandas as pd

from src.data.transform import CrossValidationSplit, CVStrategy
from sklearn import model_selection
from src.exceptions import data as data_exceptions


class TransformTest(unittest.TestCase):
    def setUp(self):
        pass

    # Test class CrossValidation
    def test_cross_validation_creates_instance_based_on_strategy(self):
        strategies = [
            (CVStrategy.k_fold, model_selection.KFold),
            (CVStrategy.shuffle_split, model_selection.ShuffleSplit),
            (CVStrategy.leave_one_out, model_selection.LeaveOneOut),
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
