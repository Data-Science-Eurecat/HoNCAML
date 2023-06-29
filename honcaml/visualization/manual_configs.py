import copy
import numpy as np
import streamlit as st
from typing import Dict
from honcaml.config.defaults.search_spaces import default_search_spaces
from honcaml.config.defaults.tuner import default_tuner
from utils import define_metrics
from constants import (names_of_models,
                       default_models,
                       model_configs_helper,
                       metrics_mode)
from extract import extract_trained_model
from load import load_trained_model


def basic_configs() -> None:
    """
    Display basic configuration elements and save the values in the
    config_file dictionary
    """
    st.markdown("**Basic Configurations**")
    col1, col2 = st.columns([1, 6])

    if "features_all" not in st.session_state:
        st.session_state["features_all"] = []

    # add problem_type selector
    st.session_state["config_file"]["global"]["problem_type"] = \
        col1.radio("Problem type", ('Regression', 'Classification')).lower()

    # add features selector
    st.session_state["config_file"]["steps"]["data"]["extract"]["features"] = \
        col2.multiselect("Features",
                         options=st.session_state["features_all"],
                         default=st.session_state["features_all"])
    st.divider()


def data_preprocess_configs() -> None:
    """
    Add input elements to set data preprocess configurations as normalization
    of the features and of the target variables
    """
    st.write("Data Preprocess")
    data_step = st.session_state["config_file"]["steps"]["data"]

    # normalize features
    col1, _, col2 = st.columns([6, .5, 1])
    features_to_normalize = \
        col1.multiselect("Features to normalize",
                         data_step["extract"]["features"])
    if len(features_to_normalize) > 0:
        with_std = col2.radio("With std (features)", (True, False))
        data_step["transform"]["normalize"]["features"] = {
            "module": "sklearn.preprocessing.StandardScaler",
            "params": {
                "with_std": with_std,
            },
            "columns": features_to_normalize
        }
    else:
        if "features" in data_step["transform"]["normalize"]:
            data_step["transform"]["normalize"].pop("features")

    # normalize target variable
    if data_step["extract"].get("target"):
        col1, _, col2 = st.columns([6, .5, 1])
        target = data_step["extract"]["target"][0]
        if col1.radio(f"Normalize target: {target}", (True, False), index=1):
            target_with_std = col2.radio("With std (target)", (True, False))
            data_step["transform"]["normalize"]["target"] = {
                "module": "sklearn.preprocessing.StandardScaler",
                "params": {
                    "with_std": target_with_std
                },
                "columns": [target]
            }
        else:
            if "target" in data_step["transform"]["normalize"]:
                data_step["transform"]["normalize"].pop("target")

    elif st.session_state["functionality"] == "Predict":
        if "target" in data_step["transform"]["normalize"]:
            data_step["transform"]["normalize"].pop("target")

    else:
        st.warning("Add datafile and select target variable first to configure"
                   " the preprocess step")

    st.divider()


def train_model_params_configs(
        model_configs: Dict, default_params: Dict) -> None:
    """
    Add input elements to set configurations to train the models

    Args:
        model_configs (Dict): configurations of the model that will be
            applied when running the app, changes by the user on the input
            elements will be updated in this dictionary
        default_params (Dict): dictionary containing the default parameters,
            values in this dictionary will not variate
    """
    for parameter, configs in default_params.items():
        method = configs["method"]
        values = configs["values"]
        output_value = ""

        col1, col2 = st.columns([7, 1])

        use_config = \
            col2.radio("Use config", ("custom", "default"),
                       key=parameter + "_use_config")

        if use_config == "custom":
            st.session_state["default_configs"][parameter] = False
        else:
            st.session_state["default_configs"][parameter] = True

        # remove parameter from the config file, default value by sklearn will
        # be used
        if st.session_state["default_configs"][parameter]:
            col1.write(parameter)
            model_configs.pop(parameter)

        # add input options to set custom values to configure the parameter
        else:
            current_value = model_configs[parameter]
            # add a multiselect to select input options when method is choice
            if method == "choice":
                output_value = \
                    col1.multiselect(parameter, values, current_value,
                                     key=parameter,
                                     max_selections=1)
                if len(output_value) == 0:
                    st.warning("You must select one value")
                else:
                    output_value = output_value[0]

            # add a slider to select input values when method is randint or
            # qrandint
            elif method in ["randint", "qrandint"]:
                min_slider = 2
                max_slider = values[1] * 3
                output_value = \
                    col1.slider(parameter, min_slider, max_slider,
                                int(current_value),
                                key=parameter)

            # add a slider to select input values when method is uniform or
            # quniform
            elif method in ["uniform", "quniform"]:
                min_slider = 0.0
                max_slider = 1.0
                output_value = \
                    col1.slider(parameter, min_slider, max_slider,
                                float(current_value), step=0.01,
                                key=parameter)

            # update the dictionary with the config file parameters
            if output_value != model_configs[parameter]:
                model_configs[parameter] = output_value


def train_model_configs() -> None:
    """
    Display the input selector to select the model to train and its specific
    configurations
    """
    st.write("Model")
    problem_type = st.session_state["config_file"]["global"]["problem_type"]

    # initialize default_configs dict, this dictionary will be used to
    # determine if we will add the configs of a feature, or we will delete it
    # to use the default values set by sklearn
    st.session_state["default_configs"] = {}

    col1, col2 = st.columns([1, 4])
    model_name = col1.radio("Model", names_of_models[problem_type].keys(),
                            index=1,  # Random Forest Regressor / Classifier
                            label_visibility="hidden")

    # initially, we set the default values
    config_model_name = names_of_models[problem_type][model_name]
    st.session_state["config_file"]["steps"]["model"]["transform"]["fit"][
        "estimator"]["module"] = config_model_name

    st.session_state["config_file"]["steps"]["model"]["transform"]["fit"][
        "estimator"]["params"] = {}
    model_configs = st.session_state["config_file"]["steps"]["model"][
        "transform"]["fit"]["estimator"]["params"]
    for param, config in default_search_spaces[problem_type][
            config_model_name].items():
        if config["method"] == "choice":
            model_configs[param] = config["values"][0]
        elif config["method"] in ["randint", "qrandint"]:
            model_configs[param] = int(np.mean(config["values"][:2]))
        elif config["method"] in ["uniform", "quniform"]:
            model_configs[param] = float(np.mean(config["values"][:2]))

    default_params = default_search_spaces[problem_type][config_model_name]

    with col2:
        train_model_params_configs(model_configs, default_params)

    st.divider()


# TODO add possibility to add custom elements to multi-selects
def benchmark_model_params_configs(
        model_name: str, model_configs: Dict, default_params: Dict) -> None:
    """
    Add input elements to set configurations to benchmark the models

    Args:
        model_name (str): a string containing the name of the model
        model_configs (Dict): configurations of the model that will be
            applied when running the app, changes by the user on the input
            elements will be updated in this dictionary
        default_params (Dict): dictionary containing the default parameters,
            values in this dictionary will not variate
    """
    for parameter, configs in default_params.items():
        method = configs["method"]
        values = configs["values"]
        output_values = ""

        if method in ["choice", "randint", "uniform"]:
            col1, col3 = st.columns([7, 1])
        elif method in ["qrandint", "quniform"]:
            col1, col2, col3 = st.columns([5.5, 1.5, 1])
        else:
            raise Exception("method not found in the known options")

        use_config = \
            col3.radio("Use config", ("custom", "default"),
                       key=model_name + "_" + parameter + "_use_config")

        if use_config == "custom":
            st.session_state["default_configs"][model_name][parameter] = False
        else:
            st.session_state["default_configs"][model_name][parameter] = True

        # remove parameter from the config file, default value by sklearn will
        # be used
        if st.session_state["default_configs"][model_name][parameter]:
            col1.write(parameter)
            model_configs.pop(parameter)

        # add input options to set custom values to config the parameter
        else:
            # add a multiselect to select input options when method is choice
            if method == "choice":
                output_values = \
                    [*col1.multiselect(parameter, values, values,
                                       help=model_configs_helper[method],
                                       key=f"{model_name}_{parameter}")]
                if len(output_values) == 0:
                    st.warning("There must be at least one value selected")

            # add a slider to select input values when method is randint
            elif method == "randint":
                min_slider = 2
                max_slider = values[1] * 3
                output_values = \
                    [*col1.slider(parameter, min_slider, max_slider, values,
                                  help=model_configs_helper[method],
                                  key=f"{model_name}_{parameter}")]

            # add a slider to select input values and a number input field to
            # select the round increment when method is qrandint
            elif method == "qrandint":
                min_slider = 2
                max_slider = values[1] * 3
                *values_slider, round_increment = values
                output_values = \
                    [*col1.slider(parameter, min_slider, max_slider,
                                  values_slider,
                                  help=model_configs_helper[method],
                                  key=f"{model_name}_{parameter}_slider")]
                round_increment = \
                    col2.number_input("Round increment of", min_value=1,
                                      value=round_increment,
                                      key=f"{model_name}_{parameter}_increm")
                output_values.append(round_increment)

            # add a slider to select input values when method is uniform
            elif method == "uniform":
                min_slider = 0.0
                max_slider = 1.0
                output_values = \
                    [*col1.slider(parameter, min_slider, max_slider,
                                  [float(val) for val in values], step=0.01,
                                  help=model_configs_helper[method],
                                  key=f"{model_name}_{parameter}")]

            # add a slider to select input values and a number input field to
            # select the round increment when method is quniform
            elif method == "quniform":
                min_slider = 0.0
                max_slider = 1.0
                *values_slider, round_increment = values
                output_values = \
                    [*col1.slider(parameter, min_slider, max_slider,
                                  [float(val) for val in values_slider],
                                  step=0.01,
                                  help=model_configs_helper[method],
                                  key=f"{model_name}_{parameter}")]
                round_increment = \
                    col2.number_input("Round increment of", min_value=0.01,
                                      value=round_increment,
                                      key=f"{model_name}_{parameter}_increm")
                output_values.append(round_increment)

            # update the dictionary with the config file parameters
            if output_values != [*values]:
                model_configs[parameter]["values"] = output_values


def benchmark_model_configs() -> None:
    """
    Display the input selectors to select the models to benchmark and their
    specific configurations
    """
    st.write("Models")
    problem_type = st.session_state["config_file"]["global"]["problem_type"]

    # initially models is set as None, we need to set it as a dict to be able
    # to add keys and values for each model
    st.session_state["config_file"]["steps"]["benchmark"]["transform"][
        "models"] = {}

    # initialize default_configs dict if it doesn't exist, this dictionary will
    # be used to determine if we will add the configs of a feature, or we will
    # delete it to use the default values set by sklearn
    if "default_configs" not in st.session_state:
        st.session_state["default_configs"] = {}

    cols_dist = [1, 4]
    for model_name in names_of_models[problem_type].keys():
        # add a key for the model name in the default configs dict if it
        # doesn't exist
        if model_name not in st.session_state["default_configs"]:
            st.session_state["default_configs"][model_name] = {}

        # display list of possible models as a checkbox (multiple choice)
        # by default only 2 models will be selected (defined in default_models)
        col1, col2 = st.container().columns(cols_dist)
        if col1.checkbox(model_name,
                         True if model_name in default_models[problem_type]
                         else False):
            # initially, we set the default values
            config_model_name = names_of_models[problem_type][model_name]
            default_params = default_search_spaces[problem_type][
                config_model_name]
            st.session_state["config_file"]["steps"]["benchmark"]["transform"][
                "models"][config_model_name] = copy.deepcopy(default_params)
            model_configs = \
                st.session_state["config_file"]["steps"]["benchmark"][
                    "transform"]["models"][config_model_name]

            # display configs input parameters
            with col2:
                with st.expander(f"{model_name} configs:"):
                    benchmark_model_params_configs(model_name, model_configs,
                                                   default_params)

    st.divider()


def cross_validation_configs() -> None:
    """
    Display different input elements corresponding to the cross validation
    configuration arguments
    """
    st.write("Cross Validation")
    # TODO: configure other strategies than kfold
    n_splits = st.number_input("Number of splits", 2, 20, 3)

    if st.session_state["functionality"] == "Benchmark":
        st.session_state["config_file"]["steps"]["benchmark"]["transform"][
            "cross_validation"] = {
            "module": "sklearn.model_selection.KFold",
            "params": {"n_splits": n_splits}
        }
    elif st.session_state["functionality"] in ["Train", "Predict"]:
        st.session_state["config_file"]["steps"]["model"]["transform"]["fit"][
            "cross_validation"] = {
            "module": "sklearn.model_selection.KFold",
            "params": {"n_splits": n_splits}
        }

    st.divider()


def tuner_configs() -> None:
    """
    Display different input elements corresponding to the tuner configuration
    arguments
    """
    # initialize the config file with the default values
    problem_type = st.session_state["config_file"]["global"]["problem_type"]
    st.session_state["config_file"]["steps"]["benchmark"]["transform"][
        "tuner"] = copy.deepcopy(default_tuner)
    config_tuner = st.session_state["config_file"]["steps"]["benchmark"][
        "transform"]["tuner"]
    default_metric = default_tuner["tune_config"]["metric"][problem_type]

    # add input fields to change default values
    st.write("Tuner")
    col1, col2 = st.columns(2)
    config_tuner["tune_config"]["num_samples"] = \
        col1.number_input("Number of samples", 2, 100, 5)

    config_tuner["tune_config"]["metric"] = \
        col2.radio("Metric", st.session_state["problem_type_metrics"],
                   index=st.session_state["problem_type_metrics"]
                   .index(default_metric))

    # automatically set the mode to min or max depending on the metric to
    # optimize
    config_tuner["tune_config"]["mode"] = \
        metrics_mode[problem_type][config_tuner["tune_config"]["metric"]]

    st.divider()


def metrics_configs() -> None:
    """
    Display a multiselect element of the possible benchmark metrics and write
    them in the session_state dict
    """
    st.write("Metrics")
    benchmark_metrics = \
        st.multiselect("Benchmark Metrics",
                       st.session_state["problem_type_metrics"],
                       default=st.session_state["problem_type_metrics"])
    if st.session_state["functionality"] == "Benchmark":
        st.session_state["config_file"]["steps"]["benchmark"]["transform"][
            "metrics"] = benchmark_metrics
    st.divider()


def manual_configs_elements() -> None:
    """
    Add the manual configuration elements when the manual option is selected
    """
    basic_configs()
    define_metrics()
    if st.session_state["configs_level"] == "Advanced":
        st.markdown("**Advanced Configurations**")
        data_preprocess_configs()

        if st.session_state["functionality"] == "Benchmark":
            benchmark_model_configs()
            metrics_configs()
            col1, col2 = st.columns([1, 2])
            with col1:
                cross_validation_configs()
            with col2:
                tuner_configs()

        elif st.session_state["functionality"] == "Train":
            train_model_configs()
            cross_validation_configs()

    if st.session_state["functionality"] == "Predict":
        uploaded_model = extract_trained_model()
        if uploaded_model:
            load_trained_model(uploaded_model)
