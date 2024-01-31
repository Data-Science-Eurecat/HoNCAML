import os
import yaml
import joblib
import pandas as pd
import streamlit as st
from constants import (data_file_path,
                       config_file_path,
                       model_results_path)
from define_config_file import add_data_filepath


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

    joblib.dump(model, os.path.join("../..", filepath))


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
    # define path to save the trained model
    most_recent_execution = \
        max(os.listdir(os.path.join('../../', model_results_path,
                                    st.session_state["current_session"])))
    filepath = os.path.join('../../', model_results_path,
                            st.session_state["current_session"],
                            most_recent_execution)

    results_filepath = os.path.abspath(filepath)

    # Temporary solution
    st.write(f"The model is saved in the following path: {results_filepath}")


def download_predictions_button(col: st.delta_generator.DeltaGenerator = st) \
        -> None:
    """
    Add button to download predictions after execution.

    Args:
        col: Defines the column where to place the button.
    """
    filepath = os.path.join(
        "../..",
        st.session_state["config_file"]["steps"]["model"]["transform"][
            "predict"]["path"]
    )
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
    with open('logs.txt', 'r') as logs_reader:
        col.download_button(label="Download logs as .txt",
                            data=logs_reader.read(),
                            file_name='logs.txt')
