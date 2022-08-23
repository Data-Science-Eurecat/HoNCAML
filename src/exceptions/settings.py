class SettingParameterDoesNotExist(Exception):
    def __init__(self, param: str):
        self.message = f'Setting param {param} does not exist.'
        super().__init__(self.message)
