class PipelineDoesNotExists(Exception):
    def __init__(self, pipeline_path):
        self.message = f'Pipeline {pipeline_path} does not exists.'
        super().__init__(self.message)
