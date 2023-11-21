import unittest
import tempfile
import shutil
import os
import yaml

from honcaml.tools import execution
from honcaml.tests import utils as test_utils


class ExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline_name = 'test_pipeline.yaml'
        self.test_dir = tempfile.mkdtemp()
        file_data = test_utils.mock_up_read_pipeline()
        self.pipeline_path = os.path.join(
            self.test_dir, f'{self.pipeline_name}')
        with open(self.pipeline_path, 'w') as f:
            yaml.dump(file_data, f)
        self.execution = execution.Execution(self.pipeline_path)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_read_pipeline_file(self):
        pipeline_content = self.execution._read_pipeline_file()
        self.assertIsInstance(pipeline_content, dict)
        self.assertDictEqual(pipeline_content, {
            'global': {
                'problem_type': 'regression',
            },
            'steps': {
                'data': {},
                'model': {},
            },
        })
