import os
import streamlit as st
from honcaml.visualization.constants import metrics_mode, logs_path
from honcaml.visualization.define_config_file import reset_config_file


def change_configs_mode() -> None:
    """
    Reset values of "submit" and  "configs_level" when changing the value of
    configs mode button.
    """
    st.session_state["submit"] = False
    st.session_state["configs_level"] = "Advanced"
    reset_config_file()


def sidebar() -> None:
    """
    Sidebar of the web page.
    """
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload your configuration file .yaml 📄 or set your "
            "configuration parameters manually\n"
            "2. Press the `Run` button and wait until the execution finishes\n"
        )
        st.divider()
        st.markdown(
            "## About\n"
            "HoNCAML(Holistic No Code Automated Machine Learning) is a tool "
            "aimed to run automated machine learning \
            pipelines for problems of different nature; main types of pipeline"
            " would be:\n"
            "1. Training the best possible model for the problem at hand\n"
            "2. Use this model to predict other instances\n\n"

            "At this moment, the following types of problems are supported:\n"
            "- Regression\n"
            "- Classification\n"
        )


def error_message() -> None:
    """
    Display an error message.
    """
    error_file_path = os.path.join(logs_path, 'errors.txt')
    with open(error_file_path) as errors_reader:
        st.error("**There was an error during the execution:**\n\n" +
                 errors_reader.read(), icon='🚨')


def define_metrics() -> None:
    """
    Define possible metrics depending on the problem type.
    """
    problem_type = st.session_state["config_file"]["global"]["problem_type"]
    st.session_state["problem_type_metrics"] = \
        list(metrics_mode[problem_type].keys())


def define_functionality_configs_level() -> None:
    """
    Define functionality when config mode is not manual
    """
    if st.session_state["config_file"]["steps"].get("benchmark"):
        st.session_state["functionality"] = "Benchmark"
        if st.session_state["config_file"]["steps"]["benchmark"] \
                .get("transform"):
            st.session_state["configs_level"] = "Advanced"
        else:
            st.session_state["configs_level"] = "Basic"

    elif st.session_state["config_file"]["steps"]["model"].get("extract"):
        st.session_state["functionality"] = "Predict"

    else:
        st.session_state["functionality"] = "Train"


def create_output_folder() -> None:
    """
    Create output folder to save the trained model or the prediction results.
    """
    if (st.session_state["functionality"] == "Predict") and \
            ("model" in st.session_state["config_file"]["steps"]):
        path_name = \
            st.session_state["config_file"]["steps"]["model"]["transform"][
                "predict"]["path"]

    else:
        return

    if not os.path.exists(path_name):
        os.mkdir(path_name)


def create_logs_folder() -> None:
    """
    Create output folder to save the trained model or the prediction results.
    """
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)


def warning(warning_type: str) -> None:
    """
    Display a warning.
    """
    if warning_type == "data_file":
        st.warning('You must upload data file', icon="⚠️")

    elif warning_type == "config_file":
        st.warning('You must provide a configuration file', icon="⚠️")

    elif warning_type == "text_area":
        st.warning("You must introduce your configurations in the text area",
                   icon="⚠️")


def check_target_datatype(problem_type) -> None:
    """
    Check if the target data type matches the problem type selected, and if it
    does not, display a warning message pointing out the datatype of the
    selected target and the selected problem type.
    """
    if st.session_state["functionality"] in ["Train", "Benchmark"]:
        data = st.session_state.get("data_uploaded", None)
        if data is not None:
            target_dtype = data[st.session_state["target"]].dtype
            if (problem_type == "regression" and target_dtype != float) or \
                    (problem_type == "classification" and target_dtype != int):
                st.warning(
                    f"The datatype of the target selected "
                    f"`{st.session_state['target']}` is `{target_dtype}`,"
                    f" and the selected problem type is `{problem_type}`",
                    icon="⚠️")
