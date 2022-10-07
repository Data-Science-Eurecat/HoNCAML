import copy
from ray import tune
from typing import Dict

from honcaml.data import transform, extract
from honcaml.exceptions import benchmark as benchmark_exceptions
from honcaml.models import trainable
from honcaml.steps import base
from honcaml.tools.startup import params, logger
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.bayesopt import BayesOptSearch

class TuneMethods:
    randint = 'randint'
    choice = 'choice'


class BenchmarkStep(base.BaseStep):
    """
    The Benchmark step class is a steps of the main pipeline. The step
    performs a model ranking by performing a hyperparameter search and model
    selection based on the user and default settings. The extract and load
    methods allow the steps to save and restore executions to/from checkpoints.

    Attributes:
        default_settings (Dict): the default settings for the steps.
        user_settings (Dict): the user defined settings for the steps.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict,
                 step_rules: Dict) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current steps.

        Args:
            default_settings (Dict): the default settings for the steps.
            user_settings (Dict): the user defined settings for the steps.
        """
        super().__init__(default_settings, user_settings, step_rules)

        # TODO: instance the Tuner class with the requested parameters
        self._tuners = None  # []ray.tune.Tuner, un tuner per model
        self._models_config = extract.read_yaml(params['models_config_path'])

        self._dataset = None

    def _extract(self, settings: Dict) -> None:
        """
        The extract process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the extract ETL process.
        """
        pass

    def _process_search_space(self, search_space: Dict) -> Dict:
        cleaned_search_space = {}
        for hyper_parameter, space in search_space.items():
            method = space['method']
            values = space['values']

            tune_method = eval(
                self._models_config['search_space_mapping'][method])

            if method == TuneMethods.randint:
                min_value, max_value = values
                cleaned_search_space[hyper_parameter] = tune_method(
                    min_value, max_value)
            elif method == TuneMethods.choice:
                cleaned_search_space[hyper_parameter] = tune_method(values)
            else:
                raise benchmark_exceptions.TuneMethodDoesNotExists(method)

        return cleaned_search_space

    def _transform(self, settings: Dict) -> None:
        """
        The transform process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the transform ETL process.
        """
        # Getting cross-validation params
        cv_split = transform.CrossValidationSplit(
            settings['cross_validation'].pop('strategy'),
            **settings.pop('cross_validation'))

        models = settings['models']
        for i, model_params in enumerate(models, start=1):
            logger.info(f'Starting search space for model {i}/{len(models)}')
            model_module = model_params['module']
            search_space = model_params['search_space']
            param_space = self._process_search_space(search_space)

            config = {
                'model_module': model_module,
                'param_space': param_space,
                'dataset': copy.deepcopy(self._dataset),
                'cv_split': copy.deepcopy(cv_split),
                'metric': settings['metric']
            }

            algo = BayesOptSearch(random_search_steps=2)

            tuner = tune.Tuner(
                trainable=trainable.EstimatorTrainer,
                # tune_config=air.RunConfig(stop={"training_iteration": 2}),
                # tune_config=tune.TuneConfig(num_samples=2, mode='min', search_alg=algo),
                tune_config=tune.TuneConfig(num_samples=2, mode='min', search_alg=algo),
                param_space=config)
            results = tuner.fit()
            print('best config: ', results.get_best_result(
                metric="score", mode="min").config)

    def _load(self, settings: Dict) -> None:
        """
        The load process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the load ETL process.
        """
        pass

    def run(self, metadata: Dict) -> Dict:
        """
        Run the benchmark step. Using a benchmark of models run the ETL
        functions to rank them and return the best one.

        Args:
            metadata (Dict): the objects output from each different previous
                step.

        Returns:
            metadata (Dict): the previous objects updated with the ones from
                the current step: the best estimator as a model from this
                library.
        """
        self._dataset = metadata['dataset']
        self.execute()

        metadata.update({
            'model_config': {
                'module': 'best_module',
                'hyperparameters': 'best_hyperparameters'
            }
        })
        return metadata
