import streamlit as st
import os
import datetime
from constants import (metrics_mode)


def set_current_session() -> str:
    """
    Creates an unique session ID.

    Returns:
        unique session ID.
    """
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def change_configs_mode() -> None:
    """
    Reset values of "submit" and  "configs_level" when changing the value of
    configs mode button.
    """
    st.session_state["submit"] = False
    st.session_state["configs_level"] = "Advanced"


def remove_previous_results() -> None:
    """
    Set session_state["submit"] as False to remove results from previous
    executions.
    """
    st.session_state["submit"] = False


def sidebar():
    """
    Sidebar of the web page
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


def error_message():
    with open('errors.txt') as errors_reader:
        st.error("**There was an error during the execution:**\n\n" +
                 errors_reader.read(), icon='🚨')


def define_metrics() -> None:
    """
    Define possible metrics depending on the problem type.
    """
    problem_type = st.session_state["config_file"]["global"]["problem_type"]
    st.session_state["metrics"] = list(metrics_mode[problem_type].keys())


def create_output_folder() -> None:
    """
    Create output folder to save the trained model or the prediction results.
    """
    if st.session_state["functionality"] == "Train":
        path_name = \
            st.session_state["config_file"]["steps"]["model"]["load"]["path"]

    elif st.session_state["functionality"] == "Predict":
        path_name = \
            st.session_state["config_file"]["steps"]["model"]["transform"][
                "predict"]["path"]

    else:
        return

    output_folder = os.path.join("../../", path_name)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)


def warning(warning_type: str) -> None:
    """
    Display a warning
    """
    if warning_type == "data_file":
        st.warning('You must upload data file', icon="⚠️")

    elif warning_type == "config_file":
        st.warning('You must provide a configuration file', icon="⚠️")

    elif warning_type == "text_area":
        st.warning("You must introduce your configurations in the text area",
                   icon="⚠️")
