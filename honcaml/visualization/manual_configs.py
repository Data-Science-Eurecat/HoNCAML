import streamlit as st
import utils

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
    st.session_state["features"] = \
        col2.multiselect("Features",
                         options=st.session_state["features_all"],
                         default=st.session_state["features_all"])
    st.divider()


def linear_svr_regressor() -> None:
    """
    Add input elements to set configurations to benchmark the Linear SVR
    regressor model
    """
    with st.expander("Linear SVR Regression configs:"):
        st.write("Fit Intercept")


def k_neighbors_regressor_configs() -> None:
    """
    Add input elements to set configurations to benchmark the K-Neighbors
    regressor model
    """
    with st.expander("K Neighbors Regression configs:"):
        st.write("Fit Intercept")


def elastic_net_configs() -> None:
    """
    Add input elements to set configurations to benchmark the Elastic Net
    model
    """
    with st.expander("Elastic Net configs:"):
        st.write("Fit Intercept")


def sgd_regressor_configs() -> None:
    """
    Add input elements to set configurations to benchmark the SDG regressor
    model
    """
    with st.expander("SGD Regressor configs:"):
        st.write("Fit Intercept")


def gradient_boosting_regressor() -> None:
    """
    Add input elements to set configurations to benchmark the Gradient Boosting
    Regressor model
    """
    with st.expander("Gradient Boosting Regressor configs:"):
        st.write("Fit Intercept")


def random_forest_configs() -> None:
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


def linear_regression_configs() -> None:
    """
    Add input elements to set configurations to benchmark the Linear Regression
    model
    """
    with st.expander("Linear Regression configs:"):
        st.write("Fit Intercept")
        st.session_state["check_true"] = st.checkbox("True", value=True)
        st.session_state["check_false"] = st.checkbox("False", value=True)


def k_neighbors_classifier_configs() -> None:
    """
    Add input elements to set configurations to benchmark the K-Neighbors
    classifier model
    """
    with st.expander("K Neighbors Classifier configs:"):
        col1, _, col2 = st.columns([5, .5, 1.5])
        col1.slider("Number of neighbors:", 1, 200, (1, 100))
        col2.write("Weights:")
        _ = col2.checkbox("Uniform", value=True)
        _ = col2.checkbox("Distance", value=True)


def logistic_regression_configs() -> None:
    """
    Add input elements to set configurations to benchmark the Logistic
    Regression model
    """
    with st.expander("Logistic Regression configs:"):
        st.write("Fit Intercept")


def sgd_classifier_configs() -> None:
    """
    Add input elements to set configurations to benchmark the SGD Classifier
    model
    """
    with st.expander("SDG Classifier configs:"):
        st.write("Fit Intercept")


def linear_svc_configs() -> None:
    """
    Add input elements to set configurations to benchmark the Linear SVC model
    """

    with st.expander("Linear SVC configs:"):
        st.write("Fit Intercept")


def gradient_boosting_classifier_configs() -> None:
    """
    Add input elements to set configurations to benchmark the Gradient Boosting
    Classifier model
    """
    with st.expander("Gradient Boosting Classifier configs:"):
        st.write("Fit Intercept")


def data_preprocess_configs() -> None:
    st.write("Data Preprocess")

    col1, _, col2 = st.columns([7, .5, 1])
    features_to_normalize = col1.multiselect("Features to normalize",
                                             st.session_state["features"])
    col2.radio("With std", ("Yes", "No"))
    st.divider()


regression_models_dict = {
    "Linear Regression": linear_regression_configs,
    "Random Forest Regressor": random_forest_configs,
    "Linear SVR": linear_svr_regressor,
    "K-Neighbors Regressor": k_neighbors_regressor_configs,
    "SGD Regressor": sgd_regressor_configs,
    "Gradient Boosting Regressor": gradient_boosting_regressor,
    "Elastic Net": elastic_net_configs
}

classification_models_dict = {
    "Logistic Regression": logistic_regression_configs,
    "Random Forest Classifier": random_forest_configs,
    "Linear SVC": linear_svc_configs,
    "K-Neighbors Classifier": k_neighbors_classifier_configs,
    "SGD Classifier": sgd_classifier_configs,
    "Gradient Boosting Classifier": gradient_boosting_classifier_configs
}


def benchmark_model_configs() -> None:
    """
    Display the input selectors to select the models to benchmark and their
    specific configurations
    """
    st.write("Models")
    if st.session_state["config_file"]["global"]["problem_type"] == \
            "regression":
        models_dict = regression_models_dict
        defaults = ("Linear Regression", "Random Forest Regressor")
    else:
        models_dict = classification_models_dict
        defaults = ("Logistic Regression", "Random Forest Classifier")

    cols_dist = [1, 4]
    for model_name, model_function in models_dict.items():
        col1, col2 = st.container().columns(cols_dist)
        if col1.checkbox(model_name,
                         True if model_name in defaults
                         else False):
            with col2:
                model_function()

    st.divider()


def fit_model_configs() -> None:
    """
    Add input selector element to allow the user choose the model desired to
    use to train
    """
    # st.write("Models")
    col1, col2 = st.columns(2)
    if st.session_state["config_file"]["global"]["problem_type"] == \
            "regression":
        models_list = list(regression_models_dict.keys())
    else:
        models_list = list(classification_models_dict.keys())

    model = col1.radio("Model", (models_list))

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
        cross_validation_configs()
        tuner_configs()
        metrics_configs()

    elif st.session_state["functionality"] == "Fit":
        fit_model_configs()
    elif st.session_state["functionality"] == "Predict":
        if st.session_state['configs_level'] == 'Advanced':
            data_preprocess_configs()
        predict_model_configs()