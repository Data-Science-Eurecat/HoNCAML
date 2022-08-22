class CVStrategyDoesNotExist(Exception):
    def __init__(self, strategy: str):
        self.message = f'Cross-validation {strategy} does not exist.'
        super().__init__(self.message)


class FileExtensionException(Exception):
    def __init__(self, extension: str):
        self.message = f'File extension {extension} is not valid.'
        super().__init__(self.message)
