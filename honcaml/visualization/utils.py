import streamlit as st
import pandas as pd
import copy
import os
import yaml
from constants import (metrics_mode,
                       data_file_path,
                       data_file_path_config_file,
                       config_file_path,
                       templates_path)


def change_configs_mode():
    """
    Reset values of "submit" and  "configs_level" when changing the value of
    configs mode button
    """
    st.session_state["submit"] = False
    st.session_state["configs_level"] = "Advanced"


def reset_config_file():
    """
    Remove config file dictionary from session state and set as false "submit"
    to remove results of previous executions
    """
    if "config_file" in st.session_state:
        st.session_state.pop("config_file")

    st.session_state["submit"] = False


def initialize_config_file():
    """

    """
    if "config_file" not in st.session_state:

        file_name = f'{st.session_state["configs_level"].lower()}_' \
                    f'{st.session_state["functionality"].lower()}.yaml'

        # add config_file key
        with open(os.path.join(templates_path, file_name), "r") as f:
            st.session_state["config_file"] = yaml.safe_load(f)

        # add data filepath
        st.session_state["config_file"]["steps"]["data"]["extract"][
            "filepath"] = data_file_path_config_file

        if "benchmark" in st.session_state["config_file"]["steps"]:
            st.session_state["config_file"]["steps"]["benchmark"]["load"][
                "path"] = \
                "honcaml_reports"
            st.session_state["config_file"]["steps"]["benchmark"]["load"][
                "save_best_config_params"] = True

        if "target" in st.session_state:
            st.session_state["config_file"]["steps"]["data"]["extract"][
                "target"] = [st.session_state["target"]]


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


def upload_data_file(data_upload_col, data_preview_container, configs_mode):
    uploaded_train_data_file = data_upload_col.file_uploader(
        "Upload your data file .csv",
        type=[".csv"],
        help="""
        **Train** dataset if the selected functionality is **Benchmark** or 
        **Fit**\n
        **Test** dataset if the selected functionality is **Predict**
        """,
        on_change=reset_config_file
    )

    if uploaded_train_data_file is not None:

        train_data = pd.read_csv(uploaded_train_data_file)
        if os.path.exists(data_file_path):
            train_data_saved = pd.read_csv(data_file_path)
            if not train_data.equals(train_data_saved):
                train_data.to_csv(data_file_path, index=False)
        else:
            train_data.to_csv(data_file_path, index=False)
        target = ""
        columns = train_data.columns.tolist()
        if (st.session_state["functionality"] != "Predict") and \
                (configs_mode == "Manually"):
            target = data_upload_col.selectbox("Target variable:", columns,
                                               key="target")
            if "config_file" in st.session_state:
                st.session_state["config_file"]["steps"]["data"]["extract"][
                    "target"] = [st.session_state["target"]]

        features = copy.deepcopy(columns)
        if target in features:
            features.remove(str(target))
        if features != st.session_state["features_all"]:
            st.session_state["features_all"] = features
        with data_preview_container.expander("Data preview"):
            st.write(train_data.head())
        return True

    else:
        return False


def download_logs_button(col):
    with open('logs.txt', 'r') as logs_reader:
        col.download_button(label="Download logs as .txt",
                            data=logs_reader.read(),
                            file_name='logs.txt')


def error_message():
    with open('errors.txt') as errors_reader:
        st.error("**There was an error during the execution:**\n\n" +
                 errors_reader.read(), icon='🚨')


def align_button(col):
    """
    Print two break lines to align buttons
    """
    col.write("\n")
    col.write("\n")


def define_metrics():
    """
    Define possible metrics depending on the problem type
    """
    problem_type = st.session_state["config_file"]["global"]["problem_type"]
    st.session_state["metrics"] = list(metrics_mode[problem_type].keys())


def write_uploaded_file(uploaded_file):
    """
    Read uploaded file, set the problem type, set the data filepath, and write
    the config file in the config_file_path
    """
    config_file = yaml.safe_load(uploaded_file)
    st.session_state["config_file"]["global"]["problem_type"] = \
        config_file["global"]["problem_type"]
    config_file["steps"]["data"]["extract"]["filepath"] = \
        st.session_state["config_file"]["steps"]["data"]["extract"]["filepath"]
    with open(config_file_path, "w") as file:
        yaml.safe_dump(config_file, file,
                       default_flow_style=False,
                       sort_keys=False)
