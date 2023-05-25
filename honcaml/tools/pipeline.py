from typing import Dict

from honcaml.exceptions import step as step_exceptions
from honcaml.steps import base as base_step, data as data_step
from honcaml.tools.startup import logger, params
from honcaml.steps import model as model_step
from honcaml.steps import benchmark as benchmark_step
from honcaml.config.defaults import models_config


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
        logger.info(f'Pipeline content {pipeline_content}')

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
            if step_name == base_step.StepType.data:
                step = data_step.DataStep(
                    params['pipeline_steps'][step_name], step_content,
                    self._pipeline_content['global'],
                    params['step_rules'][step_name])
            elif step_name == base_step.StepType.model:
                step = model_step.ModelStep(
                    params['pipeline_steps'][step_name], step_content,
                    self._pipeline_content['global'],
                    params['step_rules'][step_name])
            elif step_name == base_step.StepType.benchmark:
                step = benchmark_step.BenchmarkStep(
                    params['pipeline_steps'][step_name], step_content,
                    self._pipeline_content['global'],
                    params['step_rules'][step_name], self._execution_id,
                    models_config)
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
        # TODO: Loop the steps and check the rules defined by the settings.yaml
        #    file: params['pipeline_rules']
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
