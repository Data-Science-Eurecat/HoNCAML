class CVModuleDoesNotExist(Exception):
    def __init__(self, module: str):
        self.message = f'Cross-validation {module} does not exist.'
        super().__init__(self.message)


class FileExtensionException(Exception):
    def __init__(self, extension: str):
        self.message = f'File extension {extension} is not valid.'
        super().__init__(self.message)


class ColumnDoesNotExists(Exception):
    def __init__(self, column: str):
        self.message = f'Column {column} does not exist.'
        super().__init__(self.message)


class TargetNotSet(Exception):
    def __init__(self):
        self.message = 'Target column has not been specified.'
        super().__init__(self.message)
