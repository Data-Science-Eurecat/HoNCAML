from src.tools.step import Step
from src.tools.startup import logger
from typing import Dict


class Pipeline:
    # TODO: typed attribute (Step)
    steps = []

    def __init__(self, pipeline_content: Dict, execution_id) -> None:
        self.pipeline_content = pipeline_content
        self.execution_id = execution_id
        logger.debug(f'Pipeline content {pipeline_content}')
        # TODO: create the pipeline steps

    def run(self):
        raise NotImplementedError('This function is not implemented.')
