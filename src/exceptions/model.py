class ModelDoesNotExists(Exception):
    def __init__(self, model_name):
        self.message = f'Step {model_name} does not exist.'
        super().__init__(self.message)
