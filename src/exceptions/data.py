class CVStrategyDoesNotExist(Exception):
    def __init__(self, strategy):
        self.message = f'Cross-validation {strategy} does not exist.'
        super().__init__(self.message)
