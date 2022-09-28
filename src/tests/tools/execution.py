import unittest
from unittest.mock import patch
import tempfile
import shutil
import os
import yaml

from src.tools.startup import params
from src.tools import execution
from src.tools import utils
from src.tests import utils as test_utils
from src.exceptions import pipeline as pipeline_exception


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

    def test_get_pipeline_path(self):
        # Path exists
        self.execution._get_pipeline_path()
        self.assertEqual(self.execution.pipeline_path, os.path.join(
            self.test_dir, f'{self.pipeline_name}.yaml'))

        # Path not exists
        self.execution._pipeline_name = 'not_exists'
        with self.assertRaises(pipeline_exception.PipelineDoesNotExist):
            self.execution._get_pipeline_path()

    def test_read_pipeline_file(self):
        pipeline_content = self.execution._read_pipeline_file()
        self.assertIsInstance(pipeline_content, dict)
        self.assertDictEqual(pipeline_content, {
            'data': {},
            'model': {},
        })

    def test_parse_pipeline(self):
        pipeline_content = self.execution._parse_pipeline()
        self.assertEqual(self.execution.pipeline_path, os.path.join(
            self.test_dir, f'{self.pipeline_name}.yaml'))
        self.assertIsInstance(pipeline_content, dict)
        self.assertDictEqual(pipeline_content, {
            'data': {},
            'model': {},
        })

    def test_run(self):
        pass