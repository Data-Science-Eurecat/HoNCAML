import ray
import unittest

from honcaml.benchmark import base
from honcaml.exceptions import benchmark as exceptions


class BaseBenchmarkTest(unittest.TestCase):

    def setUp(self) -> None:
        self.recursive_space = {
            'method': 'choice',
            'values': [
                {'module': 'torch.optim.SGD',
                 'params': {
                     'lr': {
                         'method': 'loguniform',
                         'values': [0.001, 0.01]
                     },
                     'momentum': {
                         'method': 'uniform',
                         'values': [0.5, 1]
                     }
                 }
                 },
                {'module': 'torch.optim.Adam',
                 'params': {
                     'lr': {
                         'method': 'loguniform',
                         'values': [0.001, 0.01]
                     },
                     'eps': {
                         'method': 'loguniform',
                         'values': [1e-07, 1e-05]
                     }
                 }
                 }
            ]
        }

    def test_clean_parameter_search_space_randint(self):
        name = 'param'
        space = {'method': 'randint', 'values': [2, 110]}
        expected_inst = ray.tune.search.sample.Integer
        result = base.BaseBenchmark._clean_parameter_search_space(
            name, space)
        self.assertIsInstance(result[name], expected_inst)

    def test_clean_parameter_search_space_choice(self):
        name = 'param'
        space = {'method': 'choice', 'values': ['sqrt', 'log2']}
        expected_inst = ray.tune.search.sample.Categorical
        result = base.BaseBenchmark._clean_parameter_search_space(
            name, space)
        self.assertIsInstance(result[name], expected_inst)

    def test_clean_parameter_search_space_invalid(self):
        name = 'param'
        space = {'method': 'invalid', 'values': ['sqrt', 'log2']}
        with self.assertRaises(exceptions.TuneMethodDoesNotExists):
            base.BaseBenchmark._clean_parameter_search_space(name, space)

    def test_clean_parameter_search_space_recursive(self):
        name = 'param'
        space = self.recursive_space
        expected = {
            'param': ray.tune.search.sample.Categorical,
            '[param]-[torch.optim.SGD]-lr': ray.tune.search.sample.Float,
            '[param]-[torch.optim.SGD]-momentum': ray.tune.search.sample.Float,
            '[param]-[torch.optim.Adam]-lr': ray.tune.search.sample.Float,
            '[param]-[torch.optim.Adam]-eps': ray.tune.search.sample.Float
        }
        result = base.BaseBenchmark._clean_parameter_search_space(
            name, space)
        for name in expected:
            self.assertIsInstance(result[name], expected[name])

    def test_clean_internal_params_for_search_space(self):
        name = 'param'
        format_parts = '[{}]'
        join_parts = '-'
        expected = {
            '[param]-[torch.optim.SGD]-lr': ray.tune.search.sample.Float,
            '[param]-[torch.optim.SGD]-momentum': ray.tune.search.sample.Float,
            '[param]-[torch.optim.Adam]-lr': ray.tune.search.sample.Float,
            '[param]-[torch.optim.Adam]-eps': ray.tune.search.sample.Float
        }
        result = base.BaseBenchmark._clean_internal_params_for_search_space(
            name, self.recursive_space, format_parts, join_parts)
        for name in expected:
            self.assertIsInstance(result[name], expected[name])
