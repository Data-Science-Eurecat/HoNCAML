import yaml
import streamlit as st
from manual_configs import manual_configs_elements
import app_execution
import utils
import visualization as viz

# from streamlit_ttyd import terminal
# from streamlit.components.v1 import iframe
# from port_for import get_port
utils.initialize_session_state()

st.set_page_config(page_title="HoNCAML", layout="wide")

st.header("HoNCAML")

utils.sidebar()

col1, data_upload_col = st.columns(2)
configs_mode = col1.radio("Introduce configurations via:",
                          ("Manually", "Config file .yaml",
                           "Paste your configs"),
                          on_change=utils.change_configs_mode)
col1.write("")

data_preview_container = st.container()

uploaded_file = None

if configs_mode == "Manually":
    col1_1, col1_2 = col1.columns(2)
    st.session_state["configs_level"] = \
        col1_1.radio("Configurations", ("Basic", "Advanced"),
                     on_change=utils.reset_config_file)
    st.session_state["functionality"] = \
        col1_2.radio("Functionality", ("Benchmark", "Train", "Predict"),
                     on_change=utils.reset_config_file)
    st.write("")

elif configs_mode == "Config file .yaml":
    uploaded_file = col1.file_uploader(
        "Upload your configurations file .yaml",
        type=[".yaml"],
    )

elif configs_mode == "Paste your configs":
    text_area = yaml.safe_load(st.text_area("Paste here your config file in "
                                            "json or yaml format"))

# upload data file
st.session_state['data_uploaded'] = \
    utils.upload_data_file(data_upload_col,
                           data_preview_container,
                           configs_mode)

if configs_mode == "Manually":
    utils.initialize_config_file()
    manual_configs_elements()

col1, col2 = st.columns([1, 8])
button = col1.button("Run")

if button:
    if not st.session_state["data_uploaded"]:
        st.session_state["submit"] = False
        st.warning('You must upload data file', icon="⚠️")
    else:
        st.session_state["submit"] = True
        if configs_mode == "Manually":
            with col2:
                with st.spinner("Reading configs and generating configuration "
                                "file .yaml... ⏳"):
                    yaml_file = st.session_state["config_file"]
                    print(yaml_file)
                    yaml_file["steps"] = {
                        "data": yaml_file["steps"]["data"],
                        "benchmark": yaml_file["steps"]["benchmark"]
                    }
                    with open(utils.config_file_path, 'w') as file:
                        yaml.safe_dump(yaml_file, file,
                                       default_flow_style=False,
                                       sort_keys=False)
                app_execution.run()

        elif configs_mode == "Config file .yaml":
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".yaml"):
                    utils.write_uploaded_file(uploaded_file)
                    utils.read_config_file()
                    utils.define_metrics()
                    with col2:
                        app_execution.run()
                else:
                    raise ValueError("File type not supported!")
            else:
                st.warning('You must provide a configuration file', icon="⚠️")

        elif configs_mode == "Paste your configs":
            with open(utils.config_file_path, "w") as f:
                yaml.safe_dump(text_area, f,
                               default_flow_style=False, sort_keys=False)
            with col2:
                app_execution.run()


if st.session_state.get("submit"):

    col2_1, col2_2 = col2.columns([1, 4])
    if st.session_state["process_poll"] == 0:
        col2_2.success('Execution successful!', icon="✅")
        st.session_state['execution_successful'] = True
        app_execution.process_results()
        utils.download_logs_button(col2_1)
    else:
        st.session_state['execution_successful'] = False
        utils.error_message()
        utils.download_logs_button(col2_1)

    if st.session_state.get("execution_successful"):

        viz.display_best_hyperparameters()

        col1, col2 = st.columns([1, 8])
        results_display = col1.radio(
            "Display results as:", ("Table", "BarChart")
        )
        utils.align_button(col2)
        col2.download_button(
            label="Download results as .csv",
            data=st.session_state["results"].to_csv().encode('utf-8'),
            file_name='results.csv')

        viz.display_results(results_display)
