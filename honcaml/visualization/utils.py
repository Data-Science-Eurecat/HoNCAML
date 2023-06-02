import streamlit as st
import pandas as pd
import copy
import os
import yaml


def change_configs_mode():
    """
    Reset values of "submit" and  "configs_level" when changing the value of
    configs mode button
    """
    st.session_state["submit"] = False
    st.session_state["configs_level"] = "Advanced"


def initialize_session_state():
    st.session_state["regression_metrics"] = \
        ["mean_squared_error", "mean_absolute_percentage_error",
         "median_absolute_error", "r2_score", "mean_absolute_error",
         "root_mean_square_error"]
    st.session_state["classification_metrics"] = \
        ["accuracy", "precision", "sensitivity", "specificity", "f1",
         "roc_auc"]

    if "functionality" not in st.session_state:
        st.session_state["functionality"] = "Benchmark"
    if "features" not in st.session_state:
        st.session_state["features"] = []
    if "features_all" not in st.session_state:
        st.session_state["features_all"] = []
    if "benchmark_metrics" not in st.session_state:
        st.session_state["benchmark_metrics"] = []
    if "config_file" not in st.session_state:
        st.session_state["config_file"] = {
            "global": {
                "problem_type": "",
                "metrics_folder": "honcaml_reports"
            },
            "steps": {
                "data": {
                    "extract": {
                        "filepath": "data/processed/train_data.csv",
                        "target": ""
                    }
                },
                "model": {
                    "load": {
                        "path": "data/models/"
                    }
                }
            }
        }


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
        st.markdown("---")
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


def upload_data_file(data_upload_col, data_preview_container):
    uploaded_train_data_file = data_upload_col.file_uploader(
        "Upload your data file .csv",
        type=[".csv"],
        help="""
        **Train** dataset if the selected functionality is **Benchmark** or 
        **Fit**\n
        **Test** dataset if the selected functionality is **Predict**
        """
    )

    if uploaded_train_data_file is not None:

        train_data = pd.read_csv(uploaded_train_data_file)
        data_file_path = '../../data/processed/train_data.csv'
        if os.path.exists(data_file_path):
            train_data_saved = pd.read_csv(data_file_path)
            if not train_data.equals(train_data_saved):
                train_data.to_csv(data_file_path, index=False)
        else:
            train_data.to_csv(data_file_path, index=False)

        columns = train_data.columns.tolist()
        target = ''
        if st.session_state["functionality"] != "Predict":
            target = data_upload_col.selectbox("Target variable:", columns)
        features = copy.deepcopy(columns)
        if target in features:
            features.remove(str(target))
        if features != st.session_state["features_all"]:
            st.session_state["features_all"] = features
        with data_preview_container.expander("Data preview"):
            # st.write("Data preview")
            # st.write(train_data.astype(str).head()
            # .style.set_properties(**{'background-color': 'yellow'}
            # ,subset=[target]))
            st.write(train_data.head())


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
    Print two break lines to align text
    """
    col.write("\n")
    col.write("\n")


def define_metrics():
    """
    Define possible metrics depending on the problem type
    """
    if st.session_state["config_file"]["problem_type"] == "regression":
        st.session_state["metrics"] = \
            st.session_state["regression_metrics"]
    else:
        st.session_state["metrics"] = \
            st.session_state["classification_metrics"]


def write_uploaded_file(uploaded_file):
    """
    Write uploaded file
    """
    with open("../../config_file.yaml", "w") as f:
        f.write(uploaded_file.getvalue().decode("utf-8"))
        f.close()


def read_config_file():
    """
    Read config file and define the problem type in the session state
    """
    with open("../../config_file.yaml", 'r') as f:
        config_file = yaml.safe_load(f)
    st.session_state["config_file"]["problem_type"] = \
        config_file["global"]["problem_type"]