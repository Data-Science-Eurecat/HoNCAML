import unittest

from honcaml.models import general, sklearn_model
from honcaml.exceptions import model as model_exception


class GeneralTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_initialize_model(self):
        # Successful init
        model = general.initialize_model('sklearn', 'regression')
        self.assertIsInstance(model, sklearn_model.SklearnModel)

        # Invalid model type
        with self.assertRaises(model_exception.ModelDoesNotExists):
            general.initialize_model('invalid_model', 'regressor')

    def test_aggregate_cv_results(self):
        cv_results = [
            {'metric1': 1, 'metric2': 4},
            {'metric1': 2, 'metric2': 5},
            {'metric1': 3, 'metric2': 6},
        ]
        agg_results = general.aggregate_cv_results(cv_results)
        self.assertIsInstance(agg_results, dict)
        self.assertTrue('metric1' in agg_results)
        self.assertTrue('metric2' in agg_results)
        self.assertDictEqual(agg_results, {
            'metric1': 2,
            'metric2': 5,
        })
