from ray import tune
from typing import Dict, Union, Optional

from honcaml.models import general, evaluate


class EstimatorTrainer(tune.Trainable):

    def setup(self, config: Dict) -> None:
        self._model_module = config['model_module']
        self._dataset = config['dataset']
        self._cv_split = config['cv_split']
        self._param_space = config['param_space']
        self.metric = config['metric']

        self._model = general.initialize_model('sklearn', 'regressor')

        model_config = {
            'module': self._model_module,
            'hyperparameters': self._param_space,
        }
        self._model.build_model(
            model_config, self._dataset.normalization)

    def step(self):
        x, y = self._dataset.x, self._dataset.y
        cv_results = evaluate.cross_validate_model(
            self._model, x, y, self._cv_split)

        return {'score': cv_results[self.metric]}

    def save_checkpoint(
            self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        pass

    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        pass

# If reuse_actors=True, implement the following function
# def reset_config(self, new_config):
