class PipelineDoesNotExist(Exception):
    def __init__(self, pipeline_path: str):
        self.message = f'Pipeline {pipeline_path} does not exist.'
        super().__init__(self.message)


class ProblemTypeNotAllowed(Exception):
    def __init__(self, problem_type):
        self.message = f'The problem type {problem_type} is not allowed.'
        super().__init__(self.message)
