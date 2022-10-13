from ray import tune
from typing import Dict, Union, Optional

from honcaml.models import general, evaluate
from honcaml.tools import custom_typing as ct


class EstimatorTrainer(tune.Trainable):
    """
    This is a class with the aim to runs a set of experiments for search the
    best model hyperparameters configuration. In addition, it is a child class
    from ray.tune.Trainable.
    The functions to override are the following:
    - setup
    - step
    - save_checkpoint
    - load_checkpoint

    Attributes:
        _model_module (str): module of model to use.
        _dataset : dataset class instance.
        _cv_split: cross-validation object with train configurations.
        _param_space (Dict): dict with model's hyperparameters to search and
            all possible values.
        _metric (str): metric to use to evaulate the model performance.
        _model: model instance.
    """

    def setup(self, config: Dict) -> None:
        """
        Given a dict with configuration parameters to run a hyperparameter
        search for a model. The dict has to contain the following parameters:
        - model_module: module of model
        - dataset: dataset class instance
        - cv_split: cross-validation object with train configurations
        - param_space: dict with model's hyperparameters to search and all
            possible values.
        - metric (str): metric to use

        This function is invoked once training starts.

        Args:
            config (Dict): a dict with a set of configuration parameters.

        """
        self._model_module = config['model_module']
        self._dataset = config['dataset']
        self._cv_split = config['cv_split']
        self._param_space = config['param_space']
        self._metric = config['metric']
        self._problem_type = config['problem_type']

        self._model = general.initialize_model('sklearn', self._problem_type)

        model_config = {
            'module': self._model_module,
            'hyperparameters': self._param_space,
        }
        self._model.build_model(
            model_config, self._dataset.normalization)

    def step(self) -> Dict[str, ct.Number]:
        """
        This function is invoked for each iteration during the search process.
        For each iteration, it runs a cross-validation training with the
        selected hyperparameters. Furthermore, it returns the mean metrics of
        the iteration.

        Returns:
            Dict[str, ct.Number]: a dict with score of the iteration.
        """
        x, y = self._dataset.x, self._dataset.y
        cv_results = evaluate.cross_validate_model(
            self._model, x, y, self._cv_split)

        return {'score': cv_results[self._metric]}

    def save_checkpoint(
            self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        pass

    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        pass

# If reuse_actors=True, implement the following function
# def reset_config(self, new_config):
