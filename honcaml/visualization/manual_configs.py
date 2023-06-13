import streamlit as st
import utils
import copy
from honcaml.config.defaults.search_spaces import default_search_spaces
from honcaml.config.defaults.tuner import default_tuner
from typing import Dict
from constants import names_of_models, model_configs_helper, metrics_mode


def basic_configs() -> None:
    """
    Display basic configuration elements and saves the values in the
    session_state dictionary
    """
    col1, col2 = st.columns([1, 6])

    if "features_all" not in st.session_state:
        st.session_state["features_all"] = []

    st.session_state["config_file"]["global"]["problem_type"] = \
        col1.radio("Problem type", ('Regression', 'Classification')).lower()
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
    transform_data_session_state = st.session_state["config_file"]["steps"][
        "data"]["transform"]["normalize"]

    # normalize features
    col1, _, col2 = st.columns([6, .5, 1])
    features_to_normalize = \
        col1.multiselect("Features to normalize",
                         st.session_state["config_file"]["steps"]["data"]
                         ["extract"]["features"])
    if len(features_to_normalize) > 0:
        with_std = col2.radio("With std (features)", (True, False))
        transform_data_session_state["features"] = {
            "module": "sklearn.preprocessing.StandardScaler",
            "params": {
                "with_std": with_std,
            },
            "columns": features_to_normalize
        }
    else:
        if "features" in transform_data_session_state:
            transform_data_session_state.pop("features")

    # normalize target variable
    if "target" in st.session_state["config_file"]["steps"]["data"]["extract"]:
        col1, _, col2 = st.columns([6, .5, 1])
        target = st.session_state["config_file"]["steps"]["data"]["extract"][
            "target"][0]
        if col1.radio(f"Normalize target: {target}", (True, False), index=1):
            target_with_std = col2.radio("With std (target)", (True, False))
            transform_data_session_state["target"] = {
                "module": "sklearn.preprocessing.StandardScaler",
                "params": {
                    "with_std": target_with_std
                },
                "columns": [target]
            }
        else:
            if "target" in transform_data_session_state:
                transform_data_session_state.pop("target")
    else:
        st.warning("Add datafile and select target variable firss to configure"
                   " the preprocess step")

    st.divider()


# TODO add possibility to add custom elements to multi-selects
def baseline_model_configs(
        model_name: str, model_configs: Dict, default_params: Dict) -> None:
    """
    Add input elements to set configurations to benchmark the models

    Args:
        model_name (str): a string containing the name of the model
        model_configs (Dict): configurations of the model that will be
        applied when running the app, changes by the user on the input elements
        will be updated in this dictionary
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
    if problem_type == "regression":
        default_models = ("Linear Regression", "Random Forest Regressor")
    elif problem_type == "classification":
        default_models = ("Logistic Regression", "Random Forest Classifier")

    # initially models is set as None, we need to set it as a dict to be able
    # to add keys and values for each model
    if not st.session_state["config_file"]["steps"]["benchmark"]["transform"][
                "models"]:
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

        # display list of possible models as a checkbox
        # by default only 2 models will be selected
        col1, col2 = st.container().columns(cols_dist)
        if col1.checkbox(model_name,
                         True if model_name in default_models
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
                    baseline_model_configs(model_name, model_configs,
                                           default_params)

    st.divider()


def fit_model_configs() -> None:
    """
    Add input selector element to allow the user choose the model desired to
    use to train
    """
    # st.write("Models")
    col1, col2 = st.columns(2)
    problem_type = st.session_state["config_file"]["global"]["problem_type"]
    models_list = list(names_of_models[problem_type].keys())

    model = col1.radio("Model", models_list)

    col2.radio("Configs", ("model", "configs", "to", "be", "defined"))
    st.divider()


def predict_model_configs() -> None:
    """
    Add a file uploader element to input the trained model to use to predict
    accepting .sav type of files
    """
    uploaded_model = st.file_uploader(
        "Upload your trained model",
        type=[".sav"],
    )


def cross_validation_configs() -> None:
    """
    Display different input elements corresponding to the cross validation
    configuration arguments
    """
    st.write("Cross Validation")
    # TODO: configure other strategies than kfold
    n_splits = st.number_input("Number of splits", 2, 20, 3)

    st.session_state["config_file"]["steps"]["benchmark"]["transform"][
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
    col1, col2, col3 = st.columns(3)
    config_tuner["tune_config"]["num_samples"] = \
        col1.number_input("Number of samples", 2, 100, 5)
    config_tuner["tune_config"]["metric"] = \
        col2.radio("Metric", st.session_state["metrics"],
                   st.session_state["metrics"].index(default_metric))
    config_tuner["tune_config"]["time_budget_s"] = \
        col3.number_input("Time budget s", 10, 300, 120)
    # automatically set the mode to min or max depending on the metric to
    # optimize
    config_tuner["tune_config"]["mode"] = \
        metrics_mode[problem_type][config_tuner["tune_config"]["metric"]]

    st.divider()


def metrics_configs() -> None:
    """
    Display a multiselect element of the possible benchmark metrics and writes
    them in the session_state dict
    """
    st.write("Metrics")
    benchmark_metrics = st.multiselect("Benchmark Metrics",
                                       st.session_state["metrics"],
                                       default=st.session_state["metrics"])
    if "benchmark" in st.session_state["config_file"]["steps"]:
        st.session_state["config_file"]["steps"]["benchmark"]["transform"][
            "metrics"] = benchmark_metrics
    st.divider()


def manual_configs_elements() -> None:
    """
    Add the manual configuration elements when selected the manual option
    """
    st.markdown("**Basic Configurations**")
    basic_configs()
    utils.define_metrics()
    if st.session_state["configs_level"] == "Advanced" \
            and st.session_state["functionality"] == "Benchmark":
        st.markdown("**Advanced Configurations**")
        data_preprocess_configs()
        benchmark_model_configs()
        metrics_configs()
        col1, col2 = st.columns([1, 3])
        with col1:
            cross_validation_configs()
        with col2:
            tuner_configs()

    elif st.session_state["functionality"] == "Fit":
        fit_model_configs()
    elif st.session_state["functionality"] == "Predict":
        if st.session_state['configs_level'] == 'Advanced':
            data_preprocess_configs()
        predict_model_configs()
