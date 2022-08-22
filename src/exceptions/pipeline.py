class PipelineDoesNotExist(Exception):
    def __init__(self, pipeline_path: str):
        self.message = f'Pipeline {pipeline_path} does not exist.'
        super().__init__(self.message)
