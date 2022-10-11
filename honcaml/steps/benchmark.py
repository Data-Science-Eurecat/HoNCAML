import os.path

import copy
import pandas as pd
from ray import tune, air
from typing import Dict, Callable, Union

from honcaml.data import transform, extract
from honcaml.exceptions import benchmark as benchmark_exceptions
from honcaml.models import trainable
from honcaml.steps import base
from honcaml.tools import utils
from honcaml.tools.startup import params, logger


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
                 step_rules: Dict, execution_id: str) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current steps.

        Args:
            default_settings (Dict): the default settings for the steps.
            user_settings (Dict): the user defined settings for the steps.
        """
        super().__init__(default_settings, user_settings, step_rules)

        # TODO: instance the Tuner class with the requested parameters
        self._tuners = None
        self._models_config = extract.read_yaml(params['models_config_path'])
        self._store_results_folder = os.path.join(
            params['metrics_folder'], execution_id)
        self._results_per_model = {}

        self._dataset = None

        self.metric = None
        self.mode = None

    def _extract(self, settings: Dict) -> None:
        """
        The extract process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the extract ETL process.
        """
        pass

    def _clean_search_space(self, search_space: Dict) -> Dict:
        """
        Given a dict with a search space for a model, this function gets the
        module of model to import and the hyperparameters search space.
        In addition, for each hyperparameter this function gets the
        corresponding method from mapping (honcaml/config/models.yaml) to
        generate the hyperparameter values during the search.

        Args:
            search_space (Dict): a dict with

        Notes:
            The method to apply at each hyperparameter to generate the
            possible values is defined in the following link:
            https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs.

            Example of 'search_space' input parameter:
            {
                'n_estimators':
                    method: randint
                    values: [2, 110],
                  max_features:
                    method: choice
                    values: [sqrt, log2]
            }

        Returns:
            (Dict): a dict where for each hyperparameter the corresponding
            method to generate all possible values during the search.
        """
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

    @staticmethod
    def _clean_scheduler(settings: Dict) -> Union[Callable, None]:
        """
        Given a dict of settings for Tune configuration, this function checks
        if exists a configuration for 'scheduler'. If it exists, It creates a
        new scheduler instance. Otherwise, it gets None.

        Args:
            settings (Dict): a dict with Tune settings.

        Returns:
            (Callable): a scheduler instance or None.
        """
        scheduler_config = settings.get('scheduler', {})
        scheduler_module = scheduler_config.get('module', None)

        if scheduler_module is not None:
            scheduler_params = scheduler_config.get('params', {})
            scheduler = utils.import_library(
                scheduler_module, scheduler_params)
        else:
            scheduler = None

        return scheduler

    @staticmethod
    def _clean_search_algorithm(settings: Dict) -> Union[Callable, None]:
        """
        Given a dict of settings for Tune configuration, this function checks
        if exists a configuration for 'search algorithm'. If it exists,
        It creates a new search algorithm instance. Otherwise, it gets None.

        Args:
            settings (Dict): a dict with Tune settings.

        Returns:
            (Union[Callable, None]): a search algorithm instance or None.
        """
        search_algorithm_config = settings.get('search_algorithm', {})
        search_algorithm_module = search_algorithm_config.get('module', None)

        if search_algorithm_module is not None:
            search_algorithm_params = search_algorithm_config.get('params', {})
            search_algorithm = utils.import_library(
                search_algorithm_module, search_algorithm_params)
        else:
            search_algorithm = None

        return search_algorithm

    @staticmethod
    def _clean_tune_config(settings: Dict) -> Union[Dict, None]:
        """
        Given a dict with Tune settings, this function gets the parameters
        for tune.TuneConfig class.

        Args:
            settings (Dict): a dict with Tune settings.

        Returns:
            (Dict): a Tune config parameters to apply if 'tune_config' exists
            in a settings dict. Otherwise, None.
        """
        return settings.get('tune_config', None)

    @staticmethod
    def _clean_run_config(settings: Dict) -> Union[Dict, None]:
        """
        Given a dict with Tune settings, this function gets the parameters
        for tune.air.RunConfig class.

        Args:
            settings (Dict): a dict with Tune settings.

        Returns:
            (Dict): a Tune config parameters to apply if 'run_config' exists
            in a settings dict. Otherwise, None.
        """
        return settings.get('run_config', None)

    def _get_metric_and_mode(self, settings: Dict) -> None:
        """
        The metric and mode from Tune config parameters.

        Args:
            settings (Dict): a dict with Tune settings.

        """
        self.metric = self._clean_tune_config(settings)['metric']
        self.mode = self._clean_tune_config(settings)['mode']

    @staticmethod
    def _filter_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a results dataframe, this function removes useless columns.
        The columns to remove are the following:
        - 'config/cv_split',
        - 'config/dataset',

        Args:
            df (pd.DataFrame): result's dataframe.

        Returns:
            (pd.DataFrame): dataframe with selected columns.
        """
        drop_columns = [
            'config/cv_split',
            'config/dataset',
        ]
        return df.drop(columns=drop_columns)

    def _store_results(self, df: pd.DataFrame) -> None:
        """
        Given a results dataframe, this function stores the dataframe to
        disk. In addition, it sorts dataframe based in metric and mode.
        The 'mode' could be 'min' or 'max'. If it is 'min', the dataframe
        is sorted in ascending way. Otherwise, it stores the dataframe in
        descending way.

        Args:
            df (pd.DataFrame): result's dataframe.
        """
        file_path = os.path.join(self._store_results_folder, 'results.csv')
        logger.info(f'Store metrics results in {file_path}')
        # Sort score results based on if the objective is maximize or minimize
        # metric function
        ascending = True if self.mode == 'min' else False
        df \
            .sort_values(by=self.metric, ascending=ascending) \
            .to_csv(file_path, index=False)

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

        config = {
            'dataset': copy.deepcopy(self._dataset),
            'cv_split': copy.deepcopy(cv_split),
            'metric': settings['metric']
        }
        tuner_settings = settings['tuner']

        self._get_metric_and_mode(tuner_settings)
        run_config_params = self._clean_run_config(tuner_settings)
        results_df = pd.DataFrame()
        models = settings['models']
        for i, model_params in enumerate(models, start=1):
            logger.info(f'Starting search space for model {i}/{len(models)}')
            # Clean model params
            model_module = model_params['module']
            search_space = model_params['search_space']
            param_space = self._clean_search_space(search_space)

            # Adding model and model's hyper parameters
            config['model_module'] = model_module
            config['param_space'] = param_space

            # Prepare Tuner configurations
            run_config = air.RunConfig(
                name=model_module, local_dir=self._store_results_folder,
                **run_config_params)

            tune_config_params = self._clean_tune_config(tuner_settings)
            tune_config_params['scheduler'] = self._clean_scheduler(
                tuner_settings)
            tune_config_params['search_alg'] = self._clean_search_algorithm(
                tuner_settings)

            # Create Tuner and fit the models with parameters
            tuner = tune.Tuner(
                trainable=trainable.EstimatorTrainer,
                run_config=run_config,
                tune_config=tune.TuneConfig(**tune_config_params),
                param_space=config)
            results = tuner.fit()

            # Get results
            best_results = results.get_best_result(
                metric=self.metric, mode=self.mode).config
            logger.info(f'Best configuration for model {model_module} '
                        f'is {best_results}')

            self._results_per_model[model_module] = results

            iter_results_df = results.get_dataframe()
            iter_results_df = self._filter_results_dataframe(iter_results_df)

            results_df = pd.concat([results_df, iter_results_df])

        # Store results for all experiments
        self._store_results(results_df)

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
