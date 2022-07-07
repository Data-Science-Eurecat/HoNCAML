from src.tools.step import Step
from src.tools.startup import logger, params
from src.tools import utils
from src import exceptions
from typing import Dict
import importlib


class Pipeline:
    """
    The pipeline class containing the steps defined by the user. Defines the
    pipeline to be executed and runs each of the steps defined.

    Attributes:
        steps (List[step.Step]): the steps defining the pipeline.
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
        logger.debug(f'Pipeline content {pipeline_content}')
        self._setup_pipeline()

    def _setup_pipeline(self):
        """
        This function builds the pipeline structure. Using the user defined
        pipeline_content, it creates all the required steps to be executed.
        """
        utils.validate_pipeline(self.pipeline_content)
        for key in self.pipeline_content:
            if key in params['pipeline_steps']:
                step_module = params['pipeline_steps'][key]['library']
                library = '.'.join(step_module.split('.')[:-1])
                module = importlib.import_module(library)
                name = step_module.split('.')[-1]
                step = getattr(module, name)(
                    params['pipeline_steps'][key], self.pipeline_content[key])
                self.steps.append(step)
            else:
                raise exceptions.step.StepDoesNotExist(key)

    def run(self):
        raise NotImplementedError('This function is not implemented.')
