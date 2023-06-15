import os
import yaml
import streamlit as st
from subprocess import Popen
from visualization import get_results_table, create_fig_visualization
import utils


def run():
    """
    Create a subprocess to run the HoNCAML library
    """
    if "submit" not in st.session_state:
        st.session_state["submit"] = True
    with st.spinner("Running... This may take a while⏳"):
        log = open('logs.txt', 'w')
        err = open('errors.txt', 'w')
        # port = get_port((5000, 7000))
        # process = Popen(f'ttyd --port {port} --once honcaml -c config_file.yaml', shell=True)
        process = Popen(f'cd ../.. && honcaml -c config_file.yaml',
                        shell=True)
                        #, stdout=log, stderr=err)
        # process = Popen(f'ls', shell=True, stdout=log, stderr=err)
        # host = "http://localhost"
        # iframe(f"{host}:{port}", height=400)
        process.wait()
        process_poll = process.poll()
        st.session_state["process_poll"] = process_poll


def process_results():
    most_recent_execution = max(os.listdir('../../honcaml_reports'))
    st.session_state["most_recent_execution"] = most_recent_execution

    st.session_state["results"] = get_results_table(most_recent_execution)
    st.session_state["fig"] = create_fig_visualization(
        st.session_state["results"])


def generate_configs_file_yaml():
    """
    Parse the input data introduced by the user and generate the config file in
    yaml format
    """
    with st.spinner("Reading configs and generating configuration "
                    "file .yaml... ⏳"):
        yaml_file = st.session_state["config_file"]
        # make sure that the order of the steps is correct
        yaml_file["steps"] = {
            "data": yaml_file["steps"]["data"],
            "benchmark": yaml_file["steps"]["benchmark"]
        }
        with open(utils.config_file_path, 'w') as file:
            yaml.safe_dump(yaml_file, file,
                           default_flow_style=False,
                           sort_keys=False)
