import unittest

from honcaml.tools import pipeline
from honcaml.tools import utils
from honcaml.steps import data, model, benchmark
from honcaml.exceptions import step as step_exceptions


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline_content = {
            'data': {},
            'benchmark': {},
            'model': {},
        }

    def test_setup_pipeline(self):
        # Valid pipeline
        pipeline_obj = pipeline.Pipeline(
            self.pipeline_content, utils.generate_unique_id())
        self.assertEqual(len(pipeline_obj._steps), 3)
        self.assertIsInstance(pipeline_obj._steps[0], data.DataStep)
        self.assertIsInstance(pipeline_obj._steps[1], benchmark.BenchmarkStep)
        self.assertIsInstance(pipeline_obj._steps[2], model.ModelStep)

        # Invalid step
        self.pipeline_content = {
            'invalid': {},
        }
        with self.assertRaises(step_exceptions.StepDoesNotExists):
            pipeline_obj = pipeline.Pipeline(
                self.pipeline_content, utils.generate_unique_id())

    def test_validate_pipeline(self):
        pipeline_obj = pipeline.Pipeline(
            self.pipeline_content, utils.generate_unique_id())
        pipeline_obj._validate_pipeline(self.pipeline_content)
        # TODO: make assertions once implemented

    def test_run(self):
        # TODO: write test
        pass
