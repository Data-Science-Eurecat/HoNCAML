import copy
import os
import shutil

import pandas as pd
from src.frameworks import base

from honcaml.config.defaults.data_step import default_data_step
from honcaml.config.defaults.search_spaces import default_search_spaces
from honcaml.config.defaults.tuner import default_tuner
from honcaml.data.extract import read_yaml
from honcaml.models.sklearn_model import SklearnModel
from honcaml.steps import benchmark as benchmark_step, data as data_step

TMP_DATASET = '.dataset.csv'
TMP_BENCHMARK = '.honcaml'
ID = 'id'
BEST_CONF_FILE = 'best_config_params.yaml'


class HoncamlClassification(base.BaseTask):
    """
    Class to handle executions for honcaml classification tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'f1_score'
        self.global_params = {'problem_type': 'classification'}

    @staticmethod
    def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Preprocess data for the type of problem, if needed.

        Args:
            data: Input dataset.
            target: Target column name.

        Returns:
            Processed dataset, if needed.
        """
        data.to_csv(TMP_DATASET, index=None)
        global_params = {'problem_type': 'classification'}
        dataset_settings = {
            'extract': {'filepath': TMP_DATASET, 'target': target},
            'transform': None,
            'load': {'filepath': TMP_DATASET}
        }
        dataset = data_step.DataStep(
            default_data_step, dataset_settings, global_params, {}, ID)
        dataset.execute()
        data = dataset.dataset._dataset
        return data

    def search_best_model(
            self, df_train: pd.DataFrame, target: str,
            parameters: dict) -> None:
        """
        Select best model for the problem at hand and store it within the
        internal `auto_ml` attribute.

        Args:
            df_train: Training dataset.
            target: Target column name.
            parameters: General benchmark parameters.
        """
        df_train.to_csv(TMP_DATASET, index=None)
        global_params = {'problem_type': 'classification'}
        dataset_settings = {
            'extract': {'filepath': TMP_DATASET, 'target': target},
            'transform': None
        }
        tuner_settings = copy.deepcopy(default_tuner)
        tuner_settings['tune_config']['metric'] = tuner_settings[
            'tune_config']['metric'][global_params['problem_type']]
        tuner_settings['tune_config']['mode'] = tuner_settings[
            'tune_config']['mode'][global_params['problem_type']]

        benchmark_settings = {
            "transform": {
                "models": default_search_spaces[global_params['problem_type']],
                "cross_validation": {
                    "module": "sklearn.model_selection.KFold",
                    "params": {"n_splits": 3}
                },
                "metrics": [
                    "f1_score",
                ],
                "tuner": tuner_settings
            },
            "load": {
                "path": TMP_BENCHMARK,
                'save_best_config_params': True
            }
        }
        dataset = data_step.DataStep(
            default_data_step, dataset_settings, global_params, {}, ID)
        dataset.execute()
        search = benchmark_step.BenchmarkStep(
            {}, benchmark_settings, global_params, {}, ID)
        metadata = {'dataset': dataset._dataset}
        search.run(metadata)
        model_conf_file = os.path.join(TMP_BENCHMARK, ID, BEST_CONF_FILE)
        model_conf = read_yaml(model_conf_file)
        for key in list(model_conf['params']):
            if isinstance(model_conf['params'][key], float) and (
                    model_conf['params'][key] % 1 == 0.0):
                model_conf['params'][key] = int(model_conf['params'][key])
        self.automl = SklearnModel._import_estimator(model_conf)

        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values
        self.automl.fit(X_train, y_train)

        os.remove(TMP_DATASET)
        shutil.rmtree(TMP_BENCHMARK)


class HoncamlRegression(base.BaseTask):
    """
    Class to handle executions for honcaml regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()
        self.optimize_metric = 'mean_absolute_error'
        self.global_params = {'problem_type': 'regression'}

    @staticmethod
    def preprocess_data(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Preprocess data for the type of problem, if needed.

        Args:
            data: Input dataset.
            target: Target column name.

        Returns:
            Processed dataset, if needed.
        """
        data.to_csv(TMP_DATASET, index=None)
        global_params = {'problem_type': 'regression'}
        dataset_settings = {
            'extract': {'filepath': TMP_DATASET, 'target': target},
            'transform': None,
            'load': {'filepath': TMP_DATASET}
        }
        dataset = data_step.DataStep(
            default_data_step, dataset_settings, global_params, {}, ID)
        dataset.execute()
        data = dataset.dataset._dataset
        return data

    def search_best_model(
            self, df_train: pd.DataFrame, target: str,
            parameters: dict) -> None:
        """
        Select best model for the problem at hand and store it within the
        internal `auto_ml` attribute.

        Args:
            df_train: Training dataset.
            target: Target column name.
            parameters: General benchmark parameters.
        """
        df_train.to_csv(TMP_DATASET, index=None)
        global_params = {'problem_type': 'regression'}
        dataset_settings = {
            'extract': {'filepath': TMP_DATASET, 'target': target},
            'transform': None
        }
        tuner_settings = copy.deepcopy(default_tuner)
        tuner_settings['tune_config']['metric'] = tuner_settings[
            'tune_config']['metric'][global_params['problem_type']]
        tuner_settings['tune_config']['mode'] = tuner_settings[
            'tune_config']['mode'][global_params['problem_type']]

        benchmark_settings = {
            "transform": {
                "models": default_search_spaces[global_params['problem_type']],
                "cross_validation": {
                    "module": "sklearn.model_selection.KFold",
                    "params": {"n_splits": 3}
                },
                "metrics": [
                    "mean_absolute_error",
                ],
                "tuner": tuner_settings
            },
            "load": {
                "path": TMP_BENCHMARK,
                'save_best_config_params': True
            }
        }
        dataset = data_step.DataStep(
            default_data_step, dataset_settings, global_params, {}, ID)
        dataset.execute()
        search = benchmark_step.BenchmarkStep(
            {}, benchmark_settings, global_params, {}, ID)
        metadata = {'dataset': dataset._dataset}
        search.run(metadata)
        model_conf_file = os.path.join(TMP_BENCHMARK, ID, BEST_CONF_FILE)
        model_conf = read_yaml(model_conf_file)
        for key in list(model_conf['params']):
            if isinstance(model_conf['params'][key], float) and (
                    model_conf['params'][key] % 1 == 0.0):
                model_conf['params'][key] = int(model_conf['params'][key])
        self.automl = SklearnModel._import_estimator(model_conf)

        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values
        self.automl.fit(X_train, y_train)

        os.remove(TMP_DATASET)
        shutil.rmtree(TMP_BENCHMARK)
