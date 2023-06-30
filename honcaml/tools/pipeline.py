from typing import Dict

from honcaml.exceptions import step as step_exceptions
from honcaml import steps
from honcaml.tools.startup import logger, params
from honcaml.tools import utils

VALID_STEPS = ['data', 'model', 'benchmark']


class Pipeline:
    """
    The pipeline class contains the steps defined by the user. It defines the
    pipeline to be executed and runs each of the steps defined.

    Attributes:
        _steps (List[steps.Step]): Steps defining the pipeline.
        _metadata (Dict): Objects output from each step.
        _pipeline_content (Dict): Settings defining the pipeline steps.
        _execution_id (str): Execution identifier.
    """

    def __init__(self, pipeline_content: Dict, execution_id: str) -> None:
        """
        Constructor method of class. It initializes the parameters and steps
        that define the pipeline.

        Args:
            pipeline_content: Settings defining the pipeline steps and global
                parameters.
            execution_id: Execution identifier.
        """
        self._pipeline_content = pipeline_content
        logger.debug(f'Pipeline content {pipeline_content}')

        self._scope_params = [pipeline_content['global']['problem_type']]
        utils.select_scope_params(params, self._scope_params)
        self._execution_id = execution_id

        self._steps = []
        self._metadata = {}

        self._setup_pipeline()


    def _setup_pipeline(self):
        """
        Builds the pipeline structure. Using the user defined pipeline_content,
        it creates all the required steps to be executed.
        """
        self._validate_pipeline(self._pipeline_content)
        for step_name, step_content in self._pipeline_content['steps'].items():
            if step_name in VALID_STEPS:
                module = getattr(steps, step_name)
                class_name = step_name.capitalize() + 'Step'
                class_definition = getattr(module, class_name)
                step = class_definition(
                    params['steps'][step_name], step_content,
                    self._pipeline_content['global'],
                    params['step_rules'][step_name], self._execution_id)
            else:
                raise step_exceptions.StepDoesNotExists(step_name)

            self._steps.append(step)

    def _validate_pipeline(self, pipeline_content: Dict) -> None:
        """
        Validates the pipeline steps based on the rules defined to prevent
        invalid executions.

        Args:
            pipeline_content: Settings defining the pipeline steps.
        """
        # TODO: Loop the steps and check the rules
        #    Rules validation rules should be defined within
        #    global['pipeline_rules']
        #    Raise an exception when the rule validation fail

        if 'steps' not in pipeline_content:
            raise ValueError("Incorrect configuration file structure. "
                             "Missing field: steps")

        pass

    def run(self):
        """
        Run the pipeline, which means to run each step consecutively.
        """
        for i, step in enumerate(self._steps, start=1):
            logger.info(f'Running step {i}/{len(self._steps)} ...')
            self._metadata = step.run(self._metadata)
            logger.info('Step execution complete.')
