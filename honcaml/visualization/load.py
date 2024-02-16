import os
import yaml
import joblib
import pandas as pd
import streamlit as st
from honcaml.visualization.constants import (
    data_file_path, config_file_path, logs_path
)
from honcaml.visualization.define_config_file import add_data_filepath


def load_data_file() -> None:
    """
    Save data file in the specified path.
    """
    data = st.session_state["data_uploaded"]
    if os.path.exists(data_file_path):
        data_saved = pd.read_csv(data_file_path)
        if not data.equals(data_saved):
            data.to_csv(data_file_path, index=False)
    else:
        data.to_csv(data_file_path, index=False)


def load_uploaded_file() -> None:
    """
    Read uploaded config file, set the problem type, set the data filepath, and
    write the config file in the config_file_path.
    """
    st.session_state["config_file"] = \
        yaml.safe_load(st.session_state["uploaded_file"])

    # add save_best_config_params key to be able to display best parameters
    if "benchmark" in st.session_state["config_file"]["steps"]:
        st.session_state["config_file"]["steps"]["benchmark"]["load"][
            "save_best_config_params"] = True

    # add the GUI session folder to the load path
    # for benchmark
    if "benchmark" in st.session_state["config_file"]["steps"]:
        st.session_state["config_file"]["steps"]["benchmark"]["load"][
            "path"] = st.session_state["config_file"]["steps"][
                "benchmark"]["load"]["path"]

    # add data filepath
    add_data_filepath()

    with open(config_file_path, "w") as file:
        yaml.safe_dump(st.session_state["config_file"], file,
                       default_flow_style=False,
                       sort_keys=False)


def load_trained_model(uploaded_model: object) -> None:
    """
    Load updated model and save it locally

    Args:
        uploaded_model: Uploaded trained model
    """
    model = joblib.load(uploaded_model)
    filepath = st.session_state["config_file"]["steps"]["model"]["extract"][
        "filepath"]

    joblib.dump(model, os.path.join(filepath))


def load_text_area_configs():
    """
    Load config file pasted in the text area
    """
    with open(config_file_path, "w") as file:
        yaml.safe_dump(st.session_state["text_area"], file,
                       default_flow_style=False,
                       sort_keys=False)
    st.session_state["config_file"] = st.session_state["text_area"]


def download_benchmark_results_button(col: st.delta_generator.DeltaGenerator) \
        -> None:
    """
    Add button to download benchmark results after execution.

    Args:
        col: Defines the column where to place the button.
    """
    col.download_button(
        label="Download results as .csv",
        data=st.session_state["results"].to_csv().encode('utf-8'),
        file_name='results.csv')


def download_trained_model_button() -> None:
    """
    Add button to download trained model after execution.
    """
    trained_model_path = st.session_state[
        "config_file"]["steps"]["model"]["load"]["filepath"]
    # define path to save the trained model
    model = open(trained_model_path, "rb").read()
    st.download_button("Download trained model", data=model,
                       file_name="trained_model.sav")


def download_predictions_button(col: st.delta_generator.DeltaGenerator = st) \
        -> None:
    """
    Add button to download predictions after execution.

    Args:
        col: Defines the column where to place the button.
    """
    filepath = st.session_state["config_file"]["steps"]["model"]["transform"][
            "predict"]["path"]
    filename = max([file for file in os.listdir(filepath)
                    if file.startswith("predictions")])
    predictions = \
        pd.read_csv(os.path.join(filepath, filename)).to_csv(index=False) \
        .encode('utf-8')
    col.download_button(label="Download predictions as .csv",
                        data=predictions,
                        file_name='predictions.csv')


def download_logs_button(col: st.delta_generator.DeltaGenerator = st) -> None:
    """
    Add button to download execution logs.

    Args:
        col: Defines the column where to place the button.
    """

    with open(os.path.join(logs_path, 'logs.txt'), 'r') as logs_reader:
        col.download_button(label="Download logs as .txt",
                            data=logs_reader.read(),
                            file_name='logs.txt')
