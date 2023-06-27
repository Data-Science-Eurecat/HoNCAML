import os
import yaml
import streamlit as st
from constants import (data_file_path_config_file,
                       templates_path,
                       benchmark_results_path,
                       trained_model_file,
                       model_results_path)
from utils import remove_previous_results


def initialize_config_file() -> None:
    """
    Load template structure of the config file and add data filepath, results
    path, and target variable.
    """
    if "config_file" not in st.session_state:

        file_name = f'{st.session_state["configs_level"].lower()}_' \
                    f'{st.session_state["functionality"].lower()}.yaml'

        # add config_file keys
        with open(os.path.join(templates_path, file_name), "r") as f:
            st.session_state["config_file"] = yaml.safe_load(f)

        # add data filepath
        st.session_state["config_file"]["steps"]["data"]["extract"][
            "filepath"] = data_file_path_config_file

        # add results path
        if st.session_state["functionality"] == "Benchmark":
            st.session_state["config_file"]["steps"]["benchmark"]["load"][
                "path"] = os.path.join(benchmark_results_path,
                                       st.session_state["current_session"])
            # set save_best_config_params as True
            st.session_state["config_file"]["steps"]["benchmark"]["load"][
                "save_best_config_params"] = True

        elif st.session_state["functionality"] == "Train":
            st.session_state["config_file"]["steps"]["model"]["load"]["path"] \
                = os.path.join(model_results_path,
                               st.session_state["current_session"])

        elif st.session_state["functionality"] == "Predict":
            st.session_state["config_file"]["steps"]["model"]["transform"][
                "predict"]["path"] \
                = os.path.join(model_results_path,
                               st.session_state["current_session"])
            # add model filepath
            st.session_state["config_file"]["steps"]["model"]["extract"][
                "filepath"] = trained_model_file

        # add target variable
        if ("target" in st.session_state) and \
                (st.session_state["functionality"] != "Predict"):
            st.session_state["config_file"]["steps"]["data"]["extract"][
                "target"] = [st.session_state["target"]]


def reset_config_file() -> None:
    """
    When config file is removed or updated:
     - Remove config file dictionary from session state.
     - Remove results from previous executions.
    """
    if "config_file" in st.session_state:
        st.session_state.pop("config_file")

    remove_previous_results()


def reset_data_file() -> None:
    """
    When data file is removed or updated:
     - Remove target, features and transform keys from session state
     - Remove results from previous executions
    """
    data_step = st.session_state["config_file"]["steps"]["data"]
    if "target" in data_step["extract"]:
        data_step["extract"].pop("target")
    if "features" in data_step["extract"]:
        data_step["extract"].pop("features")
    st.session_state["features_all"] = []
    if "transform" in data_step:
        data_step["transform"]["normalize"]["features"]["params"]["with_std"] \
            = None
        data_step["transform"]["normalize"]["features"]["columns"] = []
        data_step["transform"]["normalize"]["target"]["params"]["with_std"] \
            = None
        data_step["transform"]["normalize"]["target"]["columns"] = []

    remove_previous_results()


def set_target_config_file() -> None:
    """
    Define target in the config_file dictionary from the target defined in the
    session_state
    """
    st.session_state["config_file"]["steps"]["data"]["extract"][
        "target"] = [st.session_state["target"]]
