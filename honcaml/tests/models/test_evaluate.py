import pandas as pd
import unittest

from honcaml.data import tabular, normalization, transform
from honcaml.models import sklearn_model, evaluate
from honcaml.tests import utils


class EvaluateTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_cross_validate_model(self):
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'params': {}}
        model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        model.build_model(model_config, norm)
        dataset = tabular.TabularDataset()
        dataset._dataset = utils.mock_up_read_dataframe()
        dataset._features = ['col1', 'col2']
        dataset._target = ['target1', 'target2']
        x, y = dataset.values
        metrics = ['mean_absolute_error', 'root_mean_squared_error']
        cv_split = transform.CrossValidationSplit(
            **{'module': 'sklearn.model_selection.KFold',
               'params': {'n_splits': 2}})
        cv_results = evaluate.cross_validate_model(
            model, x, y, cv_split, metrics)
        self.assertIsInstance(cv_results, dict)

    def test_compute_metrics(self):
        y_true = pd.Series([1, 2, 3, 4])
        y_pred = pd.Series([1.5, 1.5, 2.5, 3.5])
        metrics = ['mean_absolute_error', 'root_mean_squared_error']
        expected = {'mean_absolute_error': 0.5, 'root_mean_squared_error': 0.5}
        result = evaluate.compute_metrics(y_true, y_pred, metrics)
        self.assertDictEqual(expected, result)

    def test_compute_root_mean_squared_error_metric(self):
        y_true = pd.Series([1, 2, 3, 4])
        y_pred = pd.Series([0.5, 2.5, 2.5, 3.5])
        expected = 0.5
        result = evaluate.compute_root_mean_squared_error_metric(
            y_true, y_pred)
        self.assertEqual(expected, result)

    def test_compute_specificity_score_metric(self):
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = pd.Series([0, 1, 1, 1])
        expected = 0.5
        result = evaluate.compute_specificity_score_metric(y_true, y_pred)
        self.assertEqual(expected, result)

    def test_compute_roc_auc_score_metric_binary(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([1, 1, 0, 1])
        expected = 0.75
        result = evaluate.compute_roc_auc_score_metric(y_true, y_pred)
        self.assertEqual(expected, result)

    def test_compute_roc_auc_score_metric_multiclass(self):
        y_true = pd.Series([0, 1, 2, 1, 2, 0])
        y_pred = pd.Series([0, 1, 2, 0, 1, 2])
        expected = 0.625
        result = evaluate.compute_roc_auc_score_metric(y_true, y_pred)
        self.assertEqual(expected, result)
        