import logging
import os
import shutil
from typing import Union
import tempfile as tmp
import yaml

import pandas as pd
from numpy.random import default_rng

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from honcaml.tools import execution
from honcaml.data.extract import read_yaml
from honcaml.models.sklearn_model import SklearnModel

from frameworks.shared.callee import call_run, result, \
    measure_inference_times
from frameworks.shared.utils import Timer

log = logging.getLogger(__name__)

BENCHMARKS_CONFIG = 'frameworks/Honcaml/benchmark/data.yaml'
TMP_PATH_DATA = 'frameworks/Honcaml/Dataset/dataset.csv'
TMP_BENCHMARK = 'frameworks/Honcaml/results'
BEST_CONF_FILE = 'best_config_params.yaml'

def run(dataset, config):
    # log.info(f"\n**** HoNCAML [v{honcaml.__version__}] ****\n")

    X_train, y_train = dataset.train.X, dataset.train.y.squeeze()

    X_train = X_train
    y_train = y_train
    print(X_train)
    print(y_train)

    is_classification = config.type == 'classification'
    log.info("Running HoNCAML with {} number of cores".format(config.cores))

    # Mapping of benchmark metrics to HoNCAML metrics
    metrics_mapping = dict(
        acc='accuracy',
        auc='roc_auc_score',
        f1='f1_score',
        mae='mean_absolute_error',
        mse='mean_squared_error',
        rmse='root_mean_squared_error',
        log_loss='log_loss',
        r2='r2_score',
    )

    if config.metric in metrics_mapping:
        perf_metric = metrics_mapping[config.metric]
    else:
        if is_classification:
            perf_metric = metrics_mapping['f1']
        else:
            perf_metric = metrics_mapping['rmse']

    if perf_metric is None:
        log.warning("Performance metric %s not supported.", config.metric)

    config_file = {
                    "global": {
                        "problem_type": config.type
                    },
                    "steps": {
                        "data": {
                            "extract": {
                                "filepath": "",
                                "target": "Target",
                            },
                            "transform": {
                                "encoding": {
                                    "OHE": {},
                                },
                            },
                        },
                        "benchmark": {
                            "transform": {
                                "tuner": {
                                    "tune_config": {
                                        "metric": perf_metric}
                                },
                                "metrics": [perf_metric]
                            },
                            "load": {
                                "path": TMP_BENCHMARK
                            },
                        },
                    },
                }

    with Timer() as training:
        #  Busquem el millor model per honcaml
        dataset_h = pd.concat([X_train, y_train], axis=1)
        aml, evaluated_individual = execute_benchmark_pipeline(dataset_h,
                                                               config_file)
        aml.fit(X_train, y_train)
    log.info(f"Finished fit in {training.duration}s.")
    print(f"Finished fit in {training.duration}s.")

    def infer(data: Union[str, pd.DataFrame]):
        #  Extreure les prediccions de les dades
        data = pd.read_parquet(data) if isinstance(data, str) else data
        if is_classification:
            try:
                return aml.predict_proba(data)
            except (RuntimeError, AttributeError):
                return aml.predict(data)
        return aml.predict(data)

    #  Medir el temps de inferencia
    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(
            infer, dataset.inference_subsample_files)
        inference_times["df"] = measure_inference_times(
            infer, [
                (1, dataset.test.X[default_rng(seed=i).integers(
                    len(dataset.test.X)), :].reshape(1, -1))
                for i in range(100)
            ],
        )
        log.info(f"Finished inference time measurements.")

    # Fem la predicció
    log.info("Predicting on the test set.")

    with Timer() as predict:
        X_test, y_test = dataset.test.X, dataset.test.y.squeeze()
        predictions = aml.predict(X_test)

    probabilities = aml.predict_proba(X_test) if is_classification else None
    print("Y_test, predictions, probabilites")
    print(y_test, predictions, probabilities)

    log.info(f"Finished predict in {predict.duration}s.")
    print(f"Finished predict in {predict.duration}s.")

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=True,
                  models_count=10,  # provisional
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  inference_times=inference_times,)


def execute_benchmark_pipeline(df_train: pd.DataFrame,
                               config_file: dict) -> object:
    """
    Execute HoNCAML benchmark, which requires little tweaks:
    1. Store the dataset on disk in order for the execution to find it
    2. Read best model from execution results
    3. Parse correctly float integer parameters stored as float
    4. Instantiate model object from configuration

    Args:
        df_train: Dataframe
        config_file: dict

    Returns:
        Model object.
    """
    df_train.to_csv(TMP_PATH_DATA, index=None)
    config_file['steps']['data']['extract']['filepath'] = TMP_PATH_DATA

    with open(BENCHMARKS_CONFIG, 'w') as file:
        yaml.dump(config_file, file, default_flow_style=False, sort_keys=False)

    print(BENCHMARKS_CONFIG)
    execution_instance = execution.Execution(BENCHMARKS_CONFIG)
    execution_instance.run()

    # Necessitem el nombre de models individuals
    """evaluate_individual = 10  # TODO"""
    evaluate_individual = 10
    
    model_conf_file = os.path.join(
        TMP_BENCHMARK, execution_instance._execution_id, BEST_CONF_FILE)
    # Read best model configuration
    model_conf = read_yaml(model_conf_file)
    for key in list(model_conf['params']):
        if isinstance(model_conf['params'][key], float) and (
                model_conf['params'][key] % 1 == 0.0):
            model_conf['params'][key] = int(model_conf['params'][key])

    automl = SklearnModel._import_estimator(model_conf)
    return automl, evaluate_individual


if __name__ == '__main__':
    call_run(run)