import os
import yaml
import time
import copy
import streamlit as st
import pandas as pd
import manual_configs
import app_execution
import utils

from streamlit_ttyd import terminal
from streamlit.components.v1 import iframe
from port_for import get_port


st.set_page_config(
    page_title="HoNCAML",
    layout="wide"
)

st.header("HoNCAML")

utils.sidebar()

col1, col2 = st.columns(2)
configs_mode = col1.radio("Introduce configurations via:",
                          ("Manually", "Config file .yaml"))
col1.write("")

# upload data file
uploaded_train_data_file = col2.file_uploader(
    "Upload your train data file .csv",
    type=[".csv"],
)

features = []
st.session_state["regression_metrics"] = \
    ["mean_squared_error", "mean_absolute_percentage_error",
     "median_absolute_error", "r2_score", "mean_absolute_error",
     "root_mean_square_error"]
st.session_state["classification_metrics"] = \
    ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]

if "features" not in st.session_state:
    st.session_state["features"] = []
if "problem_type" not in st.session_state:
    st.session_state["problem_type"] = []
if "benchmark_metrics" not in st.session_state:
    st.session_state["benchmark_metrics"] = []
if "models" not in st.session_state:
    st.session_state["models"] = set()

if uploaded_train_data_file is not None:
    # select target variable
    train_data = pd.read_csv(uploaded_train_data_file)
    st.write("Data preview")
    st.write(train_data.head())
    #train_data.to_csv('train_data.csv', index=False)
    columns = train_data.columns.tolist()
    target = col2.selectbox("Target variable:", columns)
    features = copy.deepcopy(columns)
    features.remove(str(target))
    if features != st.session_state["features"]:
        st.session_state["features"] = features

uploaded_file = None
st.divider()

if configs_mode == "Manually":
    manual_configs.data_preprocess_configs()
    manual_configs.model_configs()
    manual_configs.cross_validation_configs()
    manual_configs.tuner_configs()
    manual_configs.metrics_configs()

else:
    uploaded_file = col1.file_uploader(
        "Upload your configurations file .yaml",
        type=[".yaml"],
    )

col1, col2 = st.columns([1, 8])
button = col1.button("Run")

if button:
    if configs_mode == "Config file .yaml":
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".yaml"):
                app_execution.run(uploaded_file, col2)
            else:
                raise ValueError("File type not supported!")
        else:
            st.warning('You must provide a configuration file', icon="⚠️")
    else:
        with col2:
            with st.spinner("Reading configs and generating configuration file"
                            " .yaml... ⏳"):
                time.sleep(2)
                st.write("done!")

if st.session_state.get("submit"):

    col2_1, col2_2 = col2.columns([1, 4])
    if st.session_state["process_poll"] == 0:
        col2_2.success('Execution successful!', icon="✅")
        st.session_state['execution_successful'] = True
        utils.download_logs_button(col2_1)
    else:
        st.session_state['execution_successful'] = False
        utils.error_message()
        utils.download_logs_button(col2_1)

    if st.session_state.get("execution_successful"):

        yaml_file_path = \
            f'../../honcaml_reports/{st.session_state["most_recent_execution"]}' \
            f'/best_config_params.yaml'
        with open(yaml_file_path, 'r') as stream:
            config_file = yaml.safe_load(stream)
        hyperparams = ""
        for elem, value in config_file['hyper_parameters'].items():
            hyperparams = hyperparams + '**' + elem + '**: **' + str(value) + \
                          '**, '
        hyperparams = hyperparams[:len(hyperparams) - 2]

        st.markdown(
            f"The best model is **{config_file['module']}**\n\n"
            f"And the best set of hyperparameter are: "
            f"**{hyperparams}**"
        )

        col1, col2 = st.columns([1, 8])
        results_display = col1.radio(
            "Display results as:", ("Table", "BarChart")
        )
        utils.align_button(col2)
        col2.download_button(
            label="Download results as .csv",
            data=st.session_state["results"].to_csv().encode('utf-8'),
            file_name='results.csv')

        if results_display == "Table":
            results_table = st.session_state["results"]\
                .set_index(['model', 'configs'])\
                .drop(columns=['model_configs'])
            results_table.columns = results_table.columns.str.replace('_', ' ')
            st.table(results_table)

        elif results_display == "BarChart":
            _, col2, _ = st.columns([1, 5, 1])
            col2.plotly_chart(st.session_state["fig"],
                              use_container_width=False)
