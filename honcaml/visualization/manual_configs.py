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


def k_neighbors_classifier_configs():
    with st.expander("K Neighbors Classifier configs:"):
        col1, _, col2 = st.columns([5, .5, 1.5])
        col1.slider("Number of neighbors:", 1, 200, (1, 100))
        col2.write("Weights:")
        _ = col2.checkbox("Uniform", value=True)
        _ = col2.checkbox("Distance", value=True)


def random_forest_configs():
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


def linear_regression_configs():
    with st.expander("Linear Regression configs:"):
        st.write("Fit Intercept")
        st.session_state["check_true"] = st.checkbox("True", value=True)
        st.session_state["check_false"] = st.checkbox("False", value=True)


def logistic_regression_configs():

    pass


def data_preprocess_configs():
    st.write("Data Preprocess")

    col1, _, col2 = st.columns([7, .5, 1])
    features_to_normalize = col1.multiselect("Features to normalize",
                                             st.session_state["features"])
    col2.radio("With std", ("Yes", "No"))
    st.divider()


def model_configs():
    col1, col2 = st.columns([1, 5])

    problem_type = col1.radio("Problem type", ('Regression', 'Classification'))
    if problem_type != st.session_state["problem_type"]:
        st.session_state["problem_type"] = problem_type

    models = set()
    with col2:
        st.write("Models")
        if st.session_state["problem_type"] == "Regression":
            st.session_state["metrics"] = \
                st.session_state["regression_metrics"]
            if st.checkbox("Random Forest Regressor"):
                random_forest_configs()
                models.add("sklearn.ensemble.RandomForestRegressor")
            if st.checkbox("Linear Regression"):
                linear_regression_configs()
                models.add("sklearn.linear_model.LinearRegression")
        else:
            st.session_state["metrics"] = \
                st.session_state["classification_metrics"]
            if st.checkbox("Random Forest Classifier"):
                random_forest_configs()
            if st.checkbox("K-Neighbors Classifier"):
                k_neighbors_classifier_configs()
    if models != st.session_state["models"]:
        st.session_state["models"] = models

    st.divider()


def cross_validation_configs():

    st.write("Cross Validation")
    col1, col2, _, col4, col5 = st.columns([2, 2, 1.3, 2, 2])
    col1.radio("Strategy", ("k_fold", "none"))
    col2.number_input("Number of splits", 2, 10, 2)
    col4.radio("Shuffle", ("True", "False"))
    col5.number_input("Random state:", 1, 100, 90)

    st.divider()


def tuner_configs():

    st.write("Tuner")
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1.3, .5, 1, 1])
    col1.radio("Search algorithm", ("OptunaSearch", "None"))
    col2.number_input("Number of samples", 2, 100, 5)
    col3.radio("Metric", st.session_state["metrics"])
    col4.radio("Mode", ("min", "max"))  # auto select min or max depend on the selected metric
    col5.number_input("Training iterations", 2, 10, 2)
    col6.radio("Scheduler", ("HyperBandScheduler", "None"))

    st.divider()


def metrics_configs():
    st.write("Metrics")
    benchmark_metrics = st.multiselect("Benchmark Metrics",
                                       st.session_state["metrics"],
                                       default=st.session_state["metrics"])
    if benchmark_metrics != st.session_state["benchmark_metrics"]:
        st.session_state["benchmark_metrics"] = benchmark_metrics

    st.divider()
