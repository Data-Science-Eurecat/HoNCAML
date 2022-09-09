import json


class StepDoesNotExists(Exception):
    def __init__(self, step_name):
        self.message = f'Step {step_name} does not exist.'
        super().__init__(self.message)


class StepValidationError(Exception):
    def __init__(self, errors):
        self.message = f'The step has some validation errors:\
            \n{json.dumps(errors, indent=2)}'
        super().__init__(self.message)
