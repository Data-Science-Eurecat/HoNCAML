import copy
import pandas as pd
import unittest
from unittest.mock import patch

from honcaml.steps import data, base
from honcaml.tests import utils
from honcaml.tools.startup import params


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.extract = base.StepPhase.extract
        self.transform = base.StepPhase.transform
        self.load = base.StepPhase.load

    def test_when_transform_not_contains_normalization_is_none(self):
        fake_settings = {
            'benchmark':
                {'transform': {
                    'param1': [1, 2, 3],
                    'param2': {'col1': 1}
                }}}




        empty_user_settings = {}
        step = data.DataStep(params['step_rules']['data'],
                             default_without_normalize_settings,
                             empty_user_settings)
        step._transform(step.step_settings)
        self.assertTrue(step.dataset.normalization is None)
