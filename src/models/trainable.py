from ray import tune


class Trainable(tune.Trainable):
    def setup(self, config):
        self._model = None  # base_model.BaseModel

    def step(self):
        # TODO: cross-validate model
        pass
