import ray
import unittest

from honcaml.benchmark import sklearn


class SklearnBenchmarkTest(unittest.TestCase):

    def test_clean_search_space(self):
        search_space = {
            'n_estimators': {
                'method': 'randint',
                'values': [2, 110]
            },
            'max_features': {
                'method': 'choice',
                'values': ['sqrt', 'log2']
            }
        }
        expected_inst = {
            'n_estimators': ray.tune.search.sample.Integer,
            'max_features': ray.tune.search.sample.Categorical
        }
        result = sklearn.SklearnBenchmark.clean_search_space(search_space)
        for key in result:
            self.assertIsInstance(result[key], expected_inst[key])
