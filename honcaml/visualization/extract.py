import yaml
import copy
import pandas as pd
import streamlit as st
from typing import Any, Union
from utils import change_configs_mode
from define_config_file import (reset_config_file,
                                reset_data_file,
                                set_target_config_file)
from load import load_data_file
from visualization import data_previsualization


def extract_configs_mode(col: st.delta_generator.DeltaGenerator) -> str:
    """
    Add an input radio selector to choose the configuration mode between
    manually, by uploading a .yaml config file, or pasting the configs in a
    text area.

    Args:
        col: Defines the column where to place the selector.
    Returns:
        configs_mode (str): Define the configuration mode.
    """
    configs_mode = col.radio("Introduce configurations via:",
                             ("Manually", "Config file .yaml",
                              "Paste your configs"),
                             on_change=change_configs_mode)

    return configs_mode


def extract_configs_level(col: st.delta_generator.DeltaGenerator) -> str:
    """
    Add an input radio selector to choose the configuration level between
    basic and advanced.

    Args:
        col: Defines the column where to place the selector.
    Returns:
        configs_mode (str): Define the configuration level.
    """

    configs_level = col.radio("Configurations",
                              ("Basic", "Advanced"),
                              on_change=reset_config_file)
    return configs_level


def extract_functionality(col: st.delta_generator.DeltaGenerator) -> str:
    """
    Add an input radio selector to choose the functionality between benchmark,
    train and predict.

    Args:
        col: Defines the column where to place the selector.
    Returns:
        configs_mode (str): Defines the configuration mode.
    """
    functionality = col.radio("Functionality",
                              ("Benchmark", "Train", "Predict"),
                              on_change=reset_config_file)
    return functionality


def extract_configs_file_yaml(col: st.delta_generator.DeltaGenerator) -> None:
    """
    Add a file uploader element to upload the .yaml configuration file.

    Args:
        col: Defines the column where to place the file uploader.
    """
    col.file_uploader(
        "Upload your configurations file .yaml",
        type=[".yaml"],
        key="uploaded_file"
    )


def extract_trained_model() -> object:
    """
    Add a file uploader element to input the trained model to use to predict
    accepting .sav type of files.

    Returns: Uploaded file.
    """
    uploaded_model = st.file_uploader(
        "Upload your trained model",
        type=[".sav"],
    )
    return uploaded_model


def extract_configs_file_text_area() -> Any:
    """
    Add a text area to paste the config file.

    Returns: Yaml object with the content pasted in the text area.
    """
    text_area = \
        st.text_area("Paste here your config file in json or yaml format")
    text_yaml = yaml.safe_load(text_area)
    return text_yaml


def extract_data_file(data_upload_col: st.delta_generator.DeltaGenerator) \
        -> Union[pd.DataFrame | None]:
    """
    Add data file uploader.

    Args:
        data_upload_col: Defines the column where to place the file uploader.
    Returns:
        data_uploaded (pd.DataFrame) if the user uploads a data file or None if
        not.
    """
    uploaded_data_file = data_upload_col.file_uploader(
        "Upload your data file .csv",
        type=[".csv"],
        help="""
        **Train** dataset if the selected functionality is **Benchmark** or
        **Fit**\n
        **Test** dataset if the selected functionality is **Predict**
        """,
        on_change=reset_data_file
    )
    if uploaded_data_file is not None:
        data_uploaded = pd.read_csv(uploaded_data_file)
        st.session_state["data_uploaded"] = data_uploaded
        return data_uploaded
    else:
        st.session_state["data_uploaded"] = None
        return None


def extract_target(data_upload_col: st.delta_generator.DeltaGenerator,
                   configs_mode: str) -> None:
    """
    Add target selector.

    Args:
        data_upload_col: Defines the column where to place the file uploader.
        configs_mode: Configuration mode [Manually, Config file .yaml,
            Paste your configs].
    """

    data = st.session_state["data_uploaded"]
    columns = data.columns.tolist()
    features = copy.deepcopy(columns)

    if (st.session_state["functionality"] != "Predict") and \
            (configs_mode == "Manually"):
        target = data_upload_col.selectbox("Target variable:", columns,
                                           key="target")

        if "config_file" in st.session_state:
            set_target_config_file()

        if target in features:
            features.remove(str(target))

    st.session_state["features_all"] = features


def add_init_input_elements() -> None:
    """
    Add initial input elements into the webpage: configs mode, config file
    uploader, data uploader, target selector, configs level, functionality,
    and data pre-visualization
    """
    col1, data_upload_col = st.columns(2)

    # define configs mode: Manually, Config file .yaml, or Paste your configs
    st.session_state["configs_mode"] = extract_configs_mode(col1)
    configs_mode = st.session_state["configs_mode"]
    col1.write("")

    # place the data preview container before the configs mode selector
    data_preview_container = st.container()

    # if "Manual" option selected, add radio selectors for the level of
    # configurations (basic or advanced) and the functionality (benchmark,
    # train or test
    if configs_mode == "Manually":
        col1_1, col1_2 = col1.columns(2)
        st.session_state["configs_level"] = extract_configs_level(col1_1)
        st.session_state["functionality"] = extract_functionality(col1_2)
        st.write("")

    # if "Config file .yaml" option selected, add file uploader yaml
    elif configs_mode == "Config file .yaml":
        extract_configs_file_yaml(col1)

    # if "Paste your configs" option selected, add text input area to paste or
    # write the configs
    elif configs_mode == "Paste your configs":
        st.session_state["text_area"] = extract_configs_file_text_area()

    # upload data file, add data preview collapsable
    extract_data_file(data_upload_col)
    if st.session_state.get("data_uploaded") is not None:
        # load data locally
        load_data_file()
        # add target selector
        extract_target(data_upload_col, configs_mode)
        # pre-visualize data
        data_previsualization(data_preview_container)
