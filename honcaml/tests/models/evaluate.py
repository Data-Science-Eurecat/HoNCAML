import pandas as pd
import unittest

from honcaml.models import evaluate


class EvaluateTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_compute_regression_metrics(self):
        y_true = pd.Series([1, 2, 3, 4])
        y_pred = pd.Series([0.5, 1.5, 2.5, 3.5])
        metrics = evaluate.compute_regression_metrics(y_true, y_pred)
        self.assertIsInstance(metrics, dict)
        self.assertTrue('mean_squared_error' in metrics)
        self.assertTrue('mean_absolute_percentage_error' in metrics)
        self.assertTrue('median_absolute_error' in metrics)
        self.assertTrue('r2_score' in metrics)
        self.assertTrue('mean_absolute_error' in metrics)
        self.assertTrue('root_mean_square_error' in metrics)

    def test_compute_classification_metrics(self):
        # TODO: rewrite once implemented
        y_true = pd.Series([1, 2, 3, 4])
        y_pred = pd.Series([0.5, 1.5, 2.5, 3.5])
        with self.assertRaises(NotImplementedError):
            _ = evaluate.compute_classification_metrics(y_true, y_pred)
