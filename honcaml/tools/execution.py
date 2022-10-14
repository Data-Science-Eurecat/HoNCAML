import os
from typing import Dict

from honcaml import exceptions
from honcaml.data import extract
from honcaml.tools import pipeline, utils
from honcaml.tools.startup import logger


class Execution:
    """
    Class to execute ML pipelines. First, it reads the pipeline content and
    creates a new Pipeline instance with pipeline file content.

    Attributes:
        _pipeline_config_file (str): Pipeline configuration file name.
        _execution_id (str): Execution identifier.
        _pipeline (pipeline.Pipeline): Pipeline instance to run.
    """

    def __init__(self, pipeline_config_file: str) -> None:
        """
        Constructor method of class. It initializes the parameters for running
        a pipeline.

        Args:
            pipeline_name: Pipeline name.
        """
        self._pipeline_config_file = pipeline_config_file
        self._execution_id = utils.generate_unique_id()
        logger.info(f'Execution id {self._execution_id}')

        # Parse pipeline content
        pipeline_content = self._read_pipeline_file()
        # Create a Pipeline instance
        self._pipeline = pipeline.Pipeline(
            pipeline_content, self._execution_id)

    def _read_pipeline_file(self) -> Dict:
        """
        Reads the pipeline file specified as a Python object.

        Returns:
            Pipeline content.
        """
        if not os.path.exists(self._pipeline_config_file):
            raise exceptions.pipeline.PipelineDoesNotExist(
                self._pipeline_config_file)
        else:
            return extract.read_yaml(self._pipeline_config_file)

    def run(self) -> None:
        """
        Parses the pipeline file and creates a new Pipeline instance to run.
        """
        self._pipeline.run()
