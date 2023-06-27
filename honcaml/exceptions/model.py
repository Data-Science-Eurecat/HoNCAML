class ModelDoesNotExists(Exception):
    def __init__(self, model_name):
        self.message = f'Step {model_name} does not exist.'
        super().__init__(self.message)


class EstimatorTypeNotAllowed(Exception):
    def __init__(self, estimator_type):
        self.message = f'The estimator type {estimator_type} is not allowed.'
        super().__init__(self.message)


class MetricDoesNotExist(Exception):
    def __init__(self, metric):
        self.message = f'Metric {metric} does not exist.'
        super().__init__(self.message)
