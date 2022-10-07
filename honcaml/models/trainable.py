from ray import tune
from typing import Dict, Union, Optional

from honcaml.models import general, evaluate


class EstimatorTrainer(tune.Trainable):

    # def __init__(self, *args, **kwargs) -> None:
    #     super(EstimatorTrainer, self).__init__(*args, **kwargs)
    #
    #     self._model_module = None
    #     self._cv_split = None
    #     self._dataset = None
    #
    #     self._model = None
    #     self._param_space = None

    def setup(self, config: Dict) -> None:
        self._model_module = config['model_module']
        self._dataset = config['dataset']
        self._cv_split = config['cv_split']
        self._param_space = config['param_space']
        self.metric = config['metric']

        print(f'setup {self._param_space}')

        # self._model = general.initialize_model(
        #     config.pop('model_type'), config.pop('estimator_type'))
        self._model = general.initialize_model(
            'sklearn', 'regressor')

    def step(self):
        # Create a dict with model configuration for the execution step
        model_config = {
            'module': self._model_module,
            'hyperparameters': self._param_space,
        }

        x, y = self._dataset.x, self._dataset.y
        self._model.build_model(
            model_config, self._dataset.normalization)

        cv_results = evaluate.cross_validate_model(
            self._model, x, y, self._cv_split)

        return {'score': cv_results[self.metric]}

    def save_checkpoint(
            self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        pass

    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        pass

# def setup(self, config):
#     self._dataset = config.pop('dataset')
#     self._cv_split = config.pop('cv_split')
#     self._train_settings = config.pop('train_settings', {})
#     self._test_settings = config.pop('test_settings', {})
#     # TODO: review model params/hyperparams format
#     self._model = general.initialize_model(
#         config.pop('model_type'), config.pop('estimator_type'))
#     self._model.build_model(config, {})
#
# def step(self):
#     x, y = self._dataset.values
#     metrics = general.cross_validate_model(
#         self._model, x, y, self._cv_split,
#         self._train_settings, self._test_settings)
#     return metrics

# If reuse_actors=True, implement the following function
# def reset_config(self, new_config):
