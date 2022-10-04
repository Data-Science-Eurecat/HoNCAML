from ray import tune
from src.models import general


class Trainable(tune.Trainable):
    def setup(self, config):
        self._dataset = config.pop('dataset')
        self._cv_split = config.pop('cv_split')
        self._train_settings = config.pop('train_settings', {})
        self._test_settings = config.pop('test_settings', {})
        # TODO: review model params/hyperparams format
        self._model = general.initialize_model(
            config.pop('model_type'), config.pop('estimator_type'))
        self._model.build_model(config, {})

    def step(self):
        x, y = self._dataset.values
        metrics = general.cross_validate_model(
            self._model, x, y, self._cv_split,
            self._train_settings, self._test_settings)
        return metrics

    # If reuse_actors=True, implement the following function
    # def reset_config(self, new_config):
