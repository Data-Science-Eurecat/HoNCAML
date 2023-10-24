import ray
import unittest

from honcaml.benchmark import base
from honcaml.exceptions import benchmark as exceptions


class BaseBenchmarkTest(unittest.TestCase):

    def test_clean_parameter_search_space_randint(self):
        space = {'method': 'randint', 'values': [2, 110]}
        expected_inst = ray.tune.search.sample.Integer
        returned = base.BaseBenchmark._clean_parameter_search_space(space)
        self.assertIsInstance(returned, expected_inst)

    def test_clean_parameter_search_space_choice(self):
        space = {'method': 'choice', 'values': ['sqrt', 'log2']}
        expected_inst = ray.tune.search.sample.Categorical
        returned = base.BaseBenchmark._clean_parameter_search_space(space)
        self.assertIsInstance(returned, expected_inst)

    def test_clean_parameter_search_space_invalid(self):
        space = {'method': 'invalid', 'values': ['sqrt', 'log2']}
        with self.assertRaises(exceptions.TuneMethodDoesNotExists):
            base.BaseBenchmark._clean_parameter_search_space(space)
