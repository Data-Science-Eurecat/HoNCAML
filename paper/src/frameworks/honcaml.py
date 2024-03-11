import os
import shutil

import pandas as pd
from src.frameworks import base

from honcaml.data.extract import read_yaml
from honcaml.models.sklearn_model import SklearnModel
from honcaml.tools import execution

CONFIG_PATH = 'config/honcaml'
BENCHMARKS_CONFIG = 'benchmarks'
PREPROCESS_CONFIG = 'preprocess'
TMP_DATASET = '.dataset.csv'
TMP_BENCHMARK = '.honcaml'
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

    def preprocess_data(
            self, data: pd.DataFrame, target: str,
            dataset: str) -> pd.DataFrame:
        """
        Preprocess data for the type of problem, if needed.

        Args:
            data: Input dataset.
            target: Target column name.
            dataset: Dataset name.

        Returns:
            Processed dataset, if needed.
        """
        data = global_preprocess_data(data, target, dataset)
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
        self.automl = execute_benchmark_pipeline(
            df_train, parameters['dataset'])
        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values
        self.automl.fit(X_train, y_train)


class HoncamlRegression(base.BaseTask):
    """
    Class to handle executions for honcaml regression tasks.
    """

    def __init__(self) -> None:
        """
        Constructor method of derived class.
        """
        super().__init__()

    def preprocess_data(
            self, data: pd.DataFrame, target: str,
            dataset: str) -> pd.DataFrame:
        """
        Preprocess data for the type of problem, if needed.

        Args:
            data: Input dataset.
            target: Target column name.
            dataset: Dataset name.

        Returns:
            Processed dataset, if needed.
        """
        data = global_preprocess_data(data, target, dataset)
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
        self.automl = execute_benchmark_pipeline(
            df_train, parameters['dataset'])
        X_train = df_train.drop(columns=target).values
        y_train = df_train[target].values
        self.automl.fit(X_train, y_train)


def global_preprocess_data(
        data: pd.DataFrame, target: str, dataset: str) -> pd.DataFrame:
    """
    Preprocess data for HoNCAML executions.

    Args:
        data: Input dataset.
        target: Target column name.
        dataset: Dataset name.

    Returns:
        Processed dataset.
    """
    data.to_csv(TMP_DATASET, index=None)
    config_file = os.path.join(
        CONFIG_PATH, PREPROCESS_CONFIG, dataset + '.yaml')
    execution.Execution(config_file).run()
    data = pd.read_csv(TMP_DATASET)
    return data


def execute_benchmark_pipeline(df_train: pd.DataFrame, dataset: str) -> object:
    """
    Execute HoNCAML benchmark, which requires little tweaks:
    1. Store the dataset on disk in order for the execution to find it
    2. Read best model from execution results
    3. Parse correctly float integer parameters stored as float
    4. Instantiate model object from configuration

    Args:
        df_train: Training dataset.
        dataset: Dataset name

    Returns:
        Model object.
    """
    # Prepare configuration
    df_train.to_csv(TMP_DATASET, index=None)
    config_file = os.path.join(
        CONFIG_PATH, BENCHMARKS_CONFIG, dataset + '.yaml')
    # Execute benchmark
    execution_instance = execution.Execution(config_file)
    execution_instance.run()
    model_conf_file = os.path.join(
        TMP_BENCHMARK, execution_instance._execution_id, BEST_CONF_FILE)
    # Read best model configuration
    model_conf = read_yaml(model_conf_file)
    for key in list(model_conf['params']):
        if isinstance(model_conf['params'][key], float) and (
                model_conf['params'][key] % 1 == 0.0):
            model_conf['params'][key] = int(model_conf['params'][key])
    # Instantiate model object
    automl = SklearnModel._import_estimator(model_conf)
    os.remove(TMP_DATASET)
    shutil.rmtree(TMP_BENCHMARK)
    return automl
