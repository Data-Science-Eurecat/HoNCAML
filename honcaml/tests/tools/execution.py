import unittest
import tempfile
import shutil
import os
import yaml

from honcaml.tools.startup import params
from honcaml.tools import execution
from honcaml.tests import utils as test_utils


class ExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline_name = 'test_pipeline'
        self.test_dir = tempfile.mkdtemp()
        file_data = test_utils.mock_up_read_pipeline()
        with open(os.path.join(
                self.test_dir, f'{self.pipeline_name}.yaml'), 'w') as f:
            yaml.dump(file_data, f)
        params['pipeline_folder'] = self.test_dir
        self.execution = execution.Execution(self.pipeline_name)

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

    def test_run(self):
        pass
