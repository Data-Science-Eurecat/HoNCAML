import pandas as pd
import unittest

from honcaml.tests import utils
from honcaml.models import sklearn_model, evaluate
from honcaml.data import tabular, normalization, transform


class EvaluateTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_cross_validate_model(self):
        problem_type = 'regression'
        model_config = {'module': 'sklearn.ensemble.RandomForestRegressor',
                        'hyperparameters': {}}
        model = sklearn_model.SklearnModel(problem_type)
        norm = normalization.Normalization({})
        model.build_model(model_config, norm)
        dataset = tabular.TabularDataset()
        dataset._dataset = utils.mock_up_read_dataframe()
        dataset._features = ['col1', 'col2']
        dataset._target = ['target1', 'target2']
        x, y = dataset.values
        cv_split = transform.CrossValidationSplit('k_fold', n_splits=2)
        train_settings = None
        test_settings = None
        cv_results = evaluate.cross_validate_model(
            model, x, y, cv_split, train_settings, test_settings)
        self.assertIsInstance(cv_results, dict)

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
