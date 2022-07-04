class SettingsDoesNotExist(Exception):
    def __init__(self, settings_key):
        self.message = f'The setting {settings_key} does not exist.'
        super().__init__(self.message)
