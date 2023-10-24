class TuneMethodDoesNotExists(Exception):
    def __init__(self, method_name):
        self.message = f"Tune method '{method_name} does not exist."
        super().__init__(self.message)


class IncorrectParameterConfiguration(Exception):
    def __init__(self, hyperparameter):
        self.message = (f"Benchmark hyperparameter '{hyperparameter}' "
                        "has incorrect format")
        super().__init__(self.message)


class IncorrectNumberOfBlocks(Exception):
    def __init__(self, number_blocks):
        self.message = f"Number of blocks '{number_blocks}' is incorrect."
        super().__init__(self.message)


class TorchLayerTypeDoesNotExist(Exception):
    def __init__(self, layer_type):
        self.message = f"Layer type '{layer_type}' does not exist.."
        super().__init__(self.message)
