import streamlit as st
import utils
import copy
from honcaml.config.defaults.search_spaces import default_search_spaces
from typing import Dict

max_features_help = """
    If “sqrt”, then max_features=sqrt(n_features).\n
    If “log2”, then max_features=log2(n_features).\n
    If None or 1.0, then max_features=n_features.\n
    If int, then consider max_features features at each split.\n
    If float, then max_features is a fraction and 
    max(1, int(max_features * n_features_in_)) features are considered 
    at each split.
"""


def basic_configs() -> None:
    """
    Display basic configuration elements and saves the values in the
    session_state dictionary
    """
    col1, col2 = st.columns([1, 6])

    st.session_state["config_file"]["global"]["problem_type"] = \
        col1.radio("Problem type", ('Regression', 'Classification')).lower()
    st.session_state["config_file"]["steps"]["data"]["extract"]["features"] = \
        col2.multiselect("Features",
                         options=st.session_state["features_all"],
                         default=st.session_state["features_all"])
    st.divider()


def random_forest_configs(
        model_configs: Dict, default_params: Dict) -> None:
    """
    Add input elements to set configurations to benchmark the Random Forest
    model
    """
    with st.expander("Random Forest configs:"):
        col1, col2, col3, col4 = st.columns([4, 1.8, .7, 1.3])
        configs = {}
        max_features_default = {"sqrt", "log2", "1 (n_features)"}
        if "max_features" not in st.session_state:
            st.session_state["max_features"] = max_features_default
        if "max_features_total" not in st.session_state:
            st.session_state["max_features_total"] = max_features_default

        new_num = \
            col2.number_input("Add another int or float", 2.0, 20.0, step=1.0)

        utils.align_button(col3)
        button_add_new_num = col3.button("Add")

        if button_add_new_num:
            st.session_state["max_features"].add(new_num)
            st.session_state["max_features_total"].add(new_num)

        utils.align_button(col4)
        reset_to_default_button = col4.button("Reset default")
        if reset_to_default_button:
            st.session_state["max_features_total"] = max_features_default
            st.session_state["max_features"] = max_features_default

        configs["max_features_raw"] = \
            col1.multiselect("Max features:",
                             st.session_state["max_features_total"],
                             st.session_state["max_features"],
                             help=max_features_help)

        configs["n_estimators"] = \
            st.slider("Number of estimators:", 0, 200, (50, 150))

        # st.session_state["max_features"] = set(configs["max_features_raw"])


helper = {
    "uniform": "Sample a float uniformly between min and max values selected "
               "on the slider",
    "quniform": "Sample a float uniformly between min and max values selected "
                "on the slider, rounding to increments of the value selected "
                "on the number input field",
    "randint": "Sample a integer uniformly between min (inclusive) and max "
               "(exclusive) values selected on the slider",
    "qrandint": "Sample a random uniformly between min (inclusive) and max "
                "(inclusive (!)) values selected on the slider, rounding to "
                "increments of the value selected on the number input field",
    "choice": "Sample an option uniformly from the specified choices"
}


# TODO add possibility to add custom elements to multiselects
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

        print(model_configs)

        use_config = \
            col3.radio("Use config", ("custom", "default"),
                       key=model_name+"_"+parameter+"_use_config")

        if use_config == "custom":
            st.session_state["default_configs"][model_name][parameter] = False
        else:
            st.session_state["default_configs"][model_name][parameter] = True

        if st.session_state["default_configs"][model_name][parameter]:
            col1.write(parameter)
            model_configs.pop(parameter)

        else:
            # add a multiselect to select input options when method is choice
            if method == "choice":
                output_values = \
                    [*col1.multiselect(parameter, values, values,
                                       help=helper[method],
                                       key=model_name + '_' + parameter)]
                if len(output_values) == 0:
                    st.warning("There must be at least one value selected")

            # add a slider to select input values when method is randint
            elif method == "randint":
                min_slider = 2
                max_slider = values[1] * 3
                output_values = \
                    [*col1.slider(parameter, min_slider, max_slider, values,
                                  help=helper[method],
                                  key=model_name + '_' + parameter)]

            # add a slider to select input values and a number input field to
            # select the round increment when method is qrandint
            elif method == "qrandint":
                min_slider = 2
                max_slider = values[1] * 3
                *values_slider, round_increment = values
                output_values = \
                    [*col1.slider(parameter, min_slider, max_slider,
                                  values_slider,
                                  help=helper[method],
                                  key=model_name + '_' + parameter + '_slider')]
                round_increment = \
                    col2.number_input("Round increment of", min_value=1,
                                      value=round_increment,
                                      key=model_name + '_' + parameter + '_increm')
                output_values.append(round_increment)

            # add a slider to select input values when method is uniform
            elif method == "uniform":
                min_slider = 0.0
                max_slider = 1.0
                output_values = \
                    [*col1.slider(parameter, min_slider, max_slider,
                                  [float(val) for val in values], step=0.01,
                                  help=helper[method],
                                  key=model_name + '_' + parameter)]

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
                                  help=helper[method],
                                  key=model_name + '_' + parameter)]
                round_increment = \
                    col2.number_input("Round increment of", min_value=0.01,
                                      value=round_increment,
                                      key=model_name + '_' + parameter + '_increm')
                output_values.append(round_increment)

            # update the dictionary with the config file parameters
            if output_values != [*values]:
                model_configs[parameter]["values"] = output_values


def data_preprocess_configs() -> None:
    """
    Add input elements to set data preprocess configurations as normalization
    of the features and of the target variables
    """
    st.write("Data Preprocess")

    # normalize features
    col1, _, col2 = st.columns([6, .5, 1])
    features_to_normalize = \
        col1.multiselect("Features to normalize",
                         st.session_state["config_file"]["steps"]["data"]
                         ["extract"]["features"])
    if len(features_to_normalize) > 0:
        with_std = col2.radio("With std (features)", (True, False))
        st.session_state["config_file"]["steps"]["data"]["transform"] = \
            {"normalize": {
                "features": {
                    "module": "sklearn.preprocessing.StandardScaler",
                    "module_params": {
                        "with_std": with_std
                    },
                    "columns": features_to_normalize
                }
            }}

    # normalize target variable
    col1, _, col2 = st.columns([6, .5, 1])
    target = \
        st.session_state["config_file"]["steps"]["data"]["extract"]["target"][0]
    if col1.radio(f"Normalize target: {target}", (True, False), index=1):
        target_with_std = col2.radio("With std (target)", (True, False))
        target_normalization_dict = {
            "module": "sklearn.preprocessing.StandardScaler",
            "module_params": {
                "with_std": target_with_std
            },
            "columns": [target]
        }
        if "transform" in st.session_state["config_file"]["steps"]["data"]:
            st.session_state["config_file"]["steps"]["data"]["transform"] \
                ["normalize"]["target"] = target_normalization_dict
        else:  # when no other features are normalized
            st.session_state["config_file"]["steps"]["data"]["transform"] = \
                {"normalize": {
                        "target": target_normalization_dict
                }}

    st.divider()


# dictionary containing the display name of the model and the name of the model
# to use in the configs file
names_of_models = {
    "regression": {
        "Linear Regression": "sklearn.linear_model.LinearRegression",
        "Random Forest Regressor": "sklearn.ensemble.RandomForestRegressor",
        "Linear SVR": "sklearn.svm.LinearSVR",
        "K-Neighbors Regressor": "sklearn.neighbors.KNeighborsRegressor",
        "SGD Regressor": "sklearn.linear_model.SGDRegressor",
        "Gradient Boosting Regressor":
            "sklearn.ensemble.GradientBoostingRegressor",
        "Elastic Net": "sklearn.linear_model.ElasticNet",
    },
    "classification": {
        "Logistic Regression": "sklearn.linear_model.LogisticRegression",
        "Random Forest Classifier": "sklearn.ensemble.RandomForestClassifier",
        "Linear SVC": "sklearn.svm.LinearSVC",
        "K-Neighbors Classifier": "sklearn.neighbors.KNeighborsClassifier",
        "SGD Classifier": "sklearn.linear_model.SGDClassifier",
        "Gradient Boosting Classifier":
            "sklearn.ensemble.GradientBoostingClassifier"
    }
}


def benchmark_model_configs() -> None:
    """
    Display the input selectors to select the models to benchmark and their
    specific configurations
    """
    st.write("Models")
    problem_type = st.session_state["config_file"]["global"]["problem_type"]
    if problem_type == 'regression':
        defaults = ("Linear Regression", "Random Forest Regressor")
    else:
        defaults = ("Logistic Regression", "Random Forest Classifier")

    # initialize the config file benchmark dictionary
    st.session_state["config_file"]["steps"]["benchmark"] = {
        "transform": {
            "models": {}
        }
    }
    # initialize default_configs dict if it doesn't exist
    if "default_configs" not in st.session_state:
        st.session_state["default_configs"] = {}

    cols_dist = [1, 4]
    for model_name in names_of_models[problem_type].keys():
        # add a key for the model name in the default configs dict if it
        # doesn't exist
        if model_name not in st.session_state["default_configs"]:
            st.session_state["default_configs"][model_name] = {}
        # print(st.session_state["default_configs"])

        # display configs input parameters
        col1, col2 = st.container().columns(cols_dist)
        if col1.checkbox(model_name,
                         True if model_name in defaults
                         else False):
            config_model_name = names_of_models[problem_type][model_name]
            default_params = default_search_spaces[problem_type] \
                [config_model_name]
            st.session_state["config_file"]["steps"]["benchmark"]["transform"] \
                ["models"][config_model_name] = copy.deepcopy(default_params)
            model_configs = \
                st.session_state["config_file"]["steps"]["benchmark"] \
                    ["transform"]["models"][config_model_name]
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
    col1, col2, _, col4, col5 = st.columns([2, 2, 1.3, 2, 2])
    col1.radio("Strategy", ("k_fold", "none"))
    col2.number_input("Number of splits", 2, 10, 2)
    col4.radio("Shuffle", ("True", "False"))
    col5.number_input("Random state:", 1, 100, 90)
    st.divider()


def tuner_configs() -> None:
    """
    Display different input elements corresponding to the tuner configuration
    arguments
    """
    st.write("Tuner")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1.3, .5, 1, 1])
    col1.radio("Search algorithm", ("OptunaSearch", "None"))
    col2.number_input("Number of samples", 2, 100, 5)
    col3.radio("Metric", st.session_state["metrics"])
    col4.radio("Mode", ("min", "max"))
    # auto select min or max depend on the selected metric
    col5.number_input("Training iterations", 2, 10, 2)
    col6.radio("Scheduler", ("HyperBandScheduler", "None"))
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
    if benchmark_metrics != st.session_state["benchmark_metrics"]:
        st.session_state["benchmark_metrics"] = benchmark_metrics
    if "benchmark" in st.session_state["config_file"]["steps"]:
        st.session_state["config_file"]["steps"]["benchmark"]["transform"]\
            ["metrics"] = benchmark_metrics
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
        cross_validation_configs()
        tuner_configs()

    elif st.session_state["functionality"] == "Fit":
        fit_model_configs()
    elif st.session_state["functionality"] == "Predict":
        if st.session_state['configs_level'] == 'Advanced':
            data_preprocess_configs()
        predict_model_configs()
