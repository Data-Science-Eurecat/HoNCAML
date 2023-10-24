import copy
import logging
import os.path
import pandas as pd
from ray import tune, air
from typing import Dict, Callable, Union

from honcaml import benchmark
from honcaml.data import transform, load
from honcaml.models import trainable
from honcaml.steps import base
from honcaml.tools import utils, custom_typing as ct
from honcaml.tools.startup import logger


class BenchmarkStep(base.BaseStep):
    """
    The Benchmark step class is a steps of the main pipeline. The step
    performs a model ranking by performing a hyperparameter search and model
    selection based on the user and default settings. The extract and load
    methods allow the steps to save and restore executions to/from checkpoints.

    Attributes:
        _store_results_folder (str): folder path to store results.
        _dataset: dataset class intance.
        _reported_metrics (List[str]): metrics to compute during hyper
            parameter search.
        _metric (str): metric function to optimize.
        _mode (str): maximize or minimize metric.
    """

    def __init__(self, default_settings: Dict, user_settings: Dict,
                 global_params: Dict, step_rules: Dict,
                 execution_id: str) -> None:
        """
        This is a constructor method of class. This function initializes
        the parameters and set up the current steps.

        Args:
            default_settings: the default settings for the steps.
            user_settings: the user defined settings for the steps.
            global_params: global parameters for the current pipeline.
            step_rules: Validation rules for this step.
            execution_id: Execution identifier.
        """
        super().__init__(default_settings, user_settings, global_params,
                         step_rules)
        self._execution_id = execution_id
        self._store_results_folder = None
        self._dataset = None
        self._benchmark = None
        self._reported_metrics = None
        self._metric = None
        self._mode = None
        self._best_model = None
        self._best_hyper_parameters = None

    @staticmethod
    def _retrieve_benchmark_class(name: str) -> Callable:
        """
        Retrieve corresponding benchmark class.

        Args:
            name: Estimator name

        Returns:
            Corresponding class
        """
        root = name.split('.')[0]
        module = getattr(benchmark, root)
        class_name = root.capitalize() + 'Benchmark'
        class_ = getattr(module, class_name)
        return class_

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

    def _clean_reported_metrics(self, settings: Dict) -> None:
        """
        Given a step settings, this function gets the metrics list to report.
        The metrics are a union set of the metrics specified in the transform
        step, together with the one used by tuner to select the best model.

        Args:
            settings (Dict): settings parameters with metrics to report.

        """
        tuner_set = set(utils.ensure_input_list(self._metric))
        settings_set = set(utils.ensure_input_list(settings['metrics']))
        self._reported_metrics = list(settings_set.union(tuner_set))

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
        self._metric = self._clean_tune_config(settings)['metric']
        self._mode = self._clean_tune_config(settings)['mode']

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
        return df.drop(columns=drop_columns, errors='ignore')

    def _sort_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a results dataframe, this function sorts dataframe based in
        metric and mode. The 'mode' could be 'min' or 'max'. If it is 'min',
        the dataframe is sorted in ascending way. Otherwise, it stores the
        dataframe in descending way.

        Args:
            df (pd.DataFrame): result's dataframe.

        Returns:
            df (pd.DataFrame): sorted dataframe.
        """
        ascending = True if self._mode == 'min' else False
        df = df \
            .sort_values(by=self._metric, ascending=ascending) \
            .reset_index(drop=True)

        return df

    def _set_store_results_folder(self):
        self._store_results_folder = os.path.join(
            self._step_settings['load']['path'], self._execution_id)

    def _store_results(self, df: pd.DataFrame) -> None:
        """
        Given a results dataframe, this function stores the dataframe to
        disk.

        Args:
            df (pd.DataFrame): result's dataframe.
        """
        file_path = os.path.join(self._store_results_folder, 'results.csv')
        logger.info(f'Store metrics results in {file_path}')
        df.to_csv(file_path, index=False)

    def _get_best_result(self, df: pd.DataFrame, df_dtypes: Dict) -> None:
        """
        Given a results dataframe, this function gets the best model and the
        best hyperparmeters configuration. In addition, it shows a log with
        the best results.

        Args:
            df (pd.DataFrame): results dataframe.
            df_dtypes: the dtypes of the columns in the df.
        """
        cond_param_columns = df.columns.str.contains('config')
        selected_columns = \
            df.columns[cond_param_columns].tolist() + self._reported_metrics
        best_params_df = df \
            .head(1)[selected_columns] \
            .drop(columns=['config/metric']) \
            .dropna(axis=1)

        # Convert dtypes
        params_dtypes = {key: value for key, value in df_dtypes.items()
                         if key in best_params_df.columns}
        best_params_df = best_params_df.astype(params_dtypes)
        # Rename columns
        best_params_df.columns = best_params_df.columns.str.split('/').str[-1]
        # Store to class attributes the best model and the best hyperparameters
        self._best_model = best_params_df['model_module'].values[0]
        self._best_hyper_parameters = self._get_best_hyper_parameters(
            best_params_df)

        metrics = self._get_best_metrics(best_params_df)
        logger.info(f'The best configuration is model {self._best_model} and '
                    f'the hyperparameter configuration: '
                    f'{self._best_hyper_parameters} with metrics {metrics}')

    def _get_best_hyper_parameters(
            self, df: pd.DataFrame) -> Dict[str, ct.Number]:
        """
        Given a dataframe with results of hyper parameter search, it gets
        a dict with the best hyper parameter configuration.

        Args:
            df (pd.DataFrame): a dataframe with results of search.

        Returns:
            (Dict[str, ct.Number])
        """
        columns_to_drop = ['model_module', 'problem_type',
                           'reported_metrics'] + self._reported_metrics
        best_hyper_params = df \
            .drop(columns=columns_to_drop, errors='ignore') \
            .to_dict('records')[0]

        return best_hyper_params

    def _get_best_metrics(self, df: pd.DataFrame) -> Dict[str, ct.Number]:
        """
        Given a dataframe with results of hyper parameter search, it gets
        a dict with the metrics of the best execution.

        Args:
            df (pd.DataFrame): a dataframe with results of search.

        Returns:
            (Dict[str, ct.Number]): a dict with metric name as keys and
            metric value as value.
        """
        return df[self._reported_metrics].to_dict('records')[0]

    def get_best_model_and_hyperparams_dict(self) -> Dict:
        """
        This function returns a dict with the best model module and the best
        hyperparameters of benchmark transform step.

        Returns:

        """
        best_config_dict = {
            'module': self._best_model,
            'params': self._best_hyper_parameters
        }

        return best_config_dict

    def _extract(self, settings: Dict) -> None:
        """
        The extract process from the benchmark step ETL.

        Args:
            settings (Dict): the settings defining the extract ETL process.
        """
        raise NotImplementedError('Extract function is not implemented')

    def _transform(self, settings: Dict) -> None:
        """
        The transform process from the benchmark step ETL. In this step, a set
        of models are trained in order to find the best hyperparameter
        configuration.
        For each model, it creates a Trainable instance with the model
        configurations. Furthermore, it fits different models with different
        configurations based on settings definitions. Finally, it gets the best
        model and best hyperparameters configurations.

        Notes:
            In results folder, this function saves a file (results.csv) with a
            results for each experiment.

        Args:
            settings (Dict): the settings defining the transform ETL process.
        """
        # Set store folder
        self._set_store_results_folder()

        # Getting cross-validation params
        cv_split = transform.CrossValidationSplit(
            **settings['cross_validation'])

        # Prepare Tuner configs
        tuner_settings = settings['tuner']
        self._get_metric_and_mode(tuner_settings)
        self._clean_reported_metrics(settings)
        # Prepare Run configs
        run_config_params = self._clean_run_config(tuner_settings)

        # Dict with configurations to use during search
        config = {
            'dataset': copy.deepcopy(self._dataset),
            'cv_split': copy.deepcopy(cv_split),
            'reported_metrics': self._reported_metrics,
            'metric': self._metric,
            'problem_type': self._global_params['problem_type']
        }

        # Create a Trainable for each model and run the hyper parameter seach.
        results_df = pd.DataFrame()
        results_dtypes = {}
        models = settings['models']
        for i, name in enumerate(models, start=1):
            logger.info(
                f'Starting search space for model {name} ({i}/{len(models)})')

            self._benchmark = self._retrieve_benchmark_class(name)

            # Clean model params
            search_space = models[name]
            param_space = self._benchmark._clean_search_space(search_space)

            # Adding model and model's hyper parameters
            config['model_module'] = name
            config['param_space'] = param_space

            # Prepare Tuner configurations
            run_config = air.RunConfig(
                name=name, local_dir=self._store_results_folder,
                verbose=0, **run_config_params)

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

            logs_to_silence = ['ray._private', 'ray.tune.search.optuna']
            for log_silence in logs_to_silence:
                log = logging.getLogger(log_silence)
                log.propagate = False

            results = tuner.fit()

            # Get best results for model iteration
            iter_results_df = results.get_dataframe()
            iter_results_df = self._filter_results_dataframe(iter_results_df)
            iter_results_df = self._sort_results(iter_results_df)

            # Concat with all results dataframe.
            results_df = pd.concat(
                [results_df, iter_results_df], ignore_index=True)
            # Retrieve column dtypes
            iter_dtypes = iter_results_df.filter(
                regex='config/param_space', axis=1).dtypes.to_dict()
            results_dtypes = iter_dtypes | results_dtypes

        # Sort and store results for all experiments
        results_df = self._sort_results(results_df)
        self._store_results(results_df)
        # Get best model and hyperparameters configuration.
        self._get_best_result(results_df, results_dtypes)

    def _load(self, settings: Dict) -> None:
        """
        The load process from the benchmark step ETL. This step checks if
        exists the key 'save_best_config_params'. If it exists, it stores a
        yaml file with the best model and hyperparmeters of transform step.

        Args:
            settings (Dict): the settings defining the load ETL process.
        """
        if settings.get('save_best_config_params', False):
            best_config_params = self.get_best_model_and_hyperparams_dict()
            file_path = os.path.join(
                self._store_results_folder, 'best_config_params.yaml')
            load.save_yaml(best_config_params, file_path)

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
            'model_config': self.get_best_model_and_hyperparams_dict()
        })

        return metadata
