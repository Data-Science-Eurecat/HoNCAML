class TuneMethodDoesNotExists(Exception):
    def __init__(self, method_name):
        self.message = f"Tune method '{method_name} does not exist."
        super().__init__(self.message)
