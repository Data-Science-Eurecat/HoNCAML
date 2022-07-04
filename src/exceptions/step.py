class StepDoesNotExists(Exception):
    def __init__(self, step_name):
        self.message = f'Step {step_name} does not exist.'
        super().__init__(self.message)
