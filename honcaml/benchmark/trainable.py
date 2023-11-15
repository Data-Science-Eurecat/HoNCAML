import copy
import numpy as np
from ray import tune
import re
from typing import Dict, Union, Optional

from honcaml.models import general, evaluate
from honcaml.steps.model import ModelActions
from honcaml.tools import custom_typing as ct
from honcaml.tools.startup import logger


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
        _reported_metrics (List[str]): metrics to report
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
        - metric (str): metric to use for evaluation

        This function is invoked once training starts.

        Args:
            config (Dict): a dict with a set of configuration parameters.
        """
        self._model_module = config['model_module']
        self._dataset = config['dataset']
        self._dataset._dataset = self._dataset._clean_dataset_for_model(
            self._dataset._dataset,
            [ModelActions.fit, ModelActions.predict])
        self._cv_split = config['cv_split']
        self._reported_metrics = config['reported_metrics']
        self._metric = config['metric']
        self._mode = config['mode']
        self._problem_type = config['problem_type']
        self._invalid_logic = config['invalid_logic']

        model_type = self._model_module.split('.')[0]
        self._model = general.initialize_model(model_type, self._problem_type)
        self._param_space = self._parse_param_space(config['param_space'])

    @staticmethod
    def _parse_param_space(
            space: dict, regexp: str = '^\\[(.+)\\]-\\[(.+)\\]-(.*)$') -> dict:
        """
        Parse parameter space, which consists of replacing conditional (nested)
        parameters that have been formatted in
        `BaseBenchmark._clean_internal_params_for_search_space`.

        Args:
            space: Parameter space
            regexp: Regular expression to capture internal parameters and parts

        Returns:
            Updated parameter space
        """
        new_space = copy.deepcopy(space)
        for key in space:
            match_obj = re.match(regexp, key)
            if match_obj:
                main = match_obj[1]
                module = match_obj[2]
                if 'params' not in new_space[main]:
                    new_space[main]['params'] = {}
                if space[main]['module'] == module:
                    internal_param = match_obj[3]
                    new_space[main]['params'][internal_param] = space[key]
                new_space.__delitem__(key)
        return new_space

    def step(self) -> Dict[str, ct.Number]:
        """
        This function is invoked for each iteration during the search process.
        For each iteration, it runs a cross-validation training with the
        selected hyperparameters. Furthermore, it returns the mean metrics of
        the iteration.

        Returns:
            Dict[str, ct.Number]: a dict with score of the iteration.
        """
        model_config = {
            'module': self._model_module,
            'params': self._param_space,
        }
        logger.debug(f'Model iteration number {self._iteration}')
        logger.debug(f'Trainable parameter space: {self._param_space}')
        # In case experiment has invalid parameter space, return worst value
        # This is in order to avoid unnecessary validations
        if self._invalid_logic(self._param_space):
            logger.debug('Invalidated experiment')
            if self._mode == 'max':
                val = -np.inf
            else:
                val = np.inf
            return {self._metric: val}
        else:
            self._model.build_model(
                model_config, self._dataset.normalization,
                self._dataset.features, self._dataset.values[1].ravel())
            x, y = self._dataset.x, self._dataset.y
            cv_results = evaluate.cross_validate_model(
                self._model, x, y, self._cv_split, self._reported_metrics,
                train_settings=self._param_space,
                test_settings=self._param_space)
        self._iteration += self._iteration
        return cv_results

    def save_checkpoint(
            self, checkpoint_dir: str) -> Optional[Union[str, Dict]]:
        pass

    def load_checkpoint(self, checkpoint: Union[Dict, str]):
        pass

    # If reuse_actors=True, implement the following function
    # def reset_config(self, new_config):
