from src.tools.startup import logger, params
from src.tools import utils
from src.exceptions import step as step_exceptions
from typing import Dict
from src.steps import base as base_step
from src.steps import data as data_step
import importlib


class Pipeline:
    """
    The pipeline class containing the steps defined by the user. Defines the
    pipeline to be executed and runs each of the steps defined.

    Attributes:
        steps (List[step.Step]): the steps defining the pipeline.
        objects (Dict): the objects output from each step.
        pipeline_content (Dict): the settings defining the pipeline steps.
        execution_id (str): the execution identifier.
    """

    def __init__(self, pipeline_content: Dict, execution_id: str) -> None:
        """
        This is a constructor method of class. This function initializes
        the params and steps that define the pipeline.

        Args:
            pipeline_content (Dict): the settings defining the pipeline steps.
            execution_id (str): the execution identifier.
        """
        self.pipeline_content = pipeline_content
        self.execution_id = execution_id
        self.steps = []
        self.objects = {}
        logger.debug(f'Pipeline content {pipeline_content}')
        self._setup_pipeline()

    def _setup_pipeline(self):
        """
        This function builds the pipeline structure. Using the user defined
        pipeline_content, it creates all the required steps to be executed.
        """
        utils.validate_pipeline(self.pipeline_content)
        for step_name, step_content in self.pipeline_content.items():
            if step_name == base_step.StepType.data:
                step = data_step.DataStep(
                    params['pipeline_steps'], self.pipeline_content)
            elif step_name == base_step.StepType.model:
                step = None
            else:
                raise step_exceptions.StepDoesNotExists(step_name)

            self.steps.append(step)

            """
            if key in params['pipeline_steps']:
                library = params['pipeline_steps'][key].pop('library')
                step = utils.import_library(
                    library, {'default_settings': params['pipeline_steps'][key],
                              'user_settings': self.pipeline_content[key]})
                self.steps.append(step)
            else:
                raise exceptions.step.StepDoesNotExist(key)
            """

    def run(self):
        for step in self.steps:
            self.objects = step.run(self.objects)
            i = 0
