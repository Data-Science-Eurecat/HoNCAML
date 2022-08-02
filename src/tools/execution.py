import os
from typing import Dict

from src import exceptions
from src.data import extract
from src.tools import pipeline, utils
from src.tools.startup import logger, params


class Execution:
    """
    The aim of this class is to execute ML pipelines. First, it reads the
    pipeline content and creates a new Pipeline instance with pipeline file
    content.

    Attributes:
        pipeline_name (str): pipeline name to execute
        execution_id (str): execution identifier
        pipeline (pipeline.Pipeline): pipeline instance to run
    """

    def __init__(self, pipeline_name: str) -> None:
        """
        This is a constructor method of class. This function initializes
        the params for running a pipeline.

        Args:
            pipeline_name (str): pipeline name.
        """
        self.pipeline_name = pipeline_name
        self.execution_id = utils.generate_unique_id()
        logger.info(f'Execution id {self.execution_id}')

        # Parse pipeline content
        pipeline_content = self._parse_pipeline()
        # Create a Pipeline instance
        self.pipeline = pipeline.Pipeline(pipeline_content, self.execution_id)

    def _get_pipeline_path(self) -> None:
        """
        This function generates pipeline file path concatenating the pipeline
        folder and filename. In addition, at the end of the filename it adds
        'yaml' extension. Finally, this function checks if the pipeline exists.
        If it does not exist raise a PipelineDoesNotExist exception.
        """
        filename = f'{self.pipeline_name}.yaml'
        self.pipeline_path = os.path.join(
            params['pipeline_folder'], filename)
        logger.info(f'Pipeline path {self.pipeline_path}')

        if not os.path.exists(self.pipeline_path):
            raise exceptions.pipeline.PipelineDoesNotExist(filename)

    def _read_pipeline_file(self) -> Dict:
        """
        This function reads a pipeline file as yaml file.

        Returns:
            pipeline content as dict
        """
        return extract.read_yaml(self.pipeline_path)

    def _parse_pipeline(self) -> Dict:
        """
        This function is divided in two steps. First one gets the pipeline
        full path. Second one gets the pipeline content.

        Returns:
            pipeline content as dict
        """
        self._get_pipeline_path()
        return self._read_pipeline_file()

    def run(self) -> None:
        """
        This function parse the pipeline file and creates a new Pipeline
        instance to run.
        """
        self.pipeline.run()
