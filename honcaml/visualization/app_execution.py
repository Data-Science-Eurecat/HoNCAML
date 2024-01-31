import os
import yaml
import streamlit as st
from subprocess import Popen
from visualization import get_results_table, create_fig_visualization
from constants import benchmark_results_path, config_file_path, logs_path


def run(col: st.delta_generator.DeltaGenerator) -> None:
    """
    Create a subprocess to run the HoNCAML library.

    Args:
        col: Defines the column where to place the spinner.
    """
    if "submit" not in st.session_state:
        st.session_state["submit"] = True
    with col:
        with st.spinner("Running... This may take a while⏳"):
            log = open(os.path.join("../../", logs_path, 
                                    st.session_state["current_session"], 
                                    'logs.txt'), 'w')
            err = open(os.path.join("../../", logs_path, 
                                    st.session_state["current_session"], 
                                    'errors.txt'), 'w')
            # port = get_port((5000, 7000))
            # process = Popen(f'ttyd --port {port} --once honcaml -c
            # config_file.yaml', shell=True)
            process = Popen('cd ../.. && honcaml -c config_file.yaml',
                            shell=True, stdout=log, stderr=err)
            # process = Popen(f'ls', shell=True, stdout=log, stderr=er
            # r)
            # host = "http://localhost"
            # iframe(f"{host}:{port}", height=400)
            process.wait()
            process_poll = process.poll()
            st.session_state["process_poll"] = process_poll


def process_results() -> None:
    """
    Find the most recent execution, create a table and a figure with the
    results of the execution
    """
    most_recent_execution = \
        max(os.listdir(os.path.join('../../', benchmark_results_path,
                       st.session_state["current_session"])))
    st.session_state["most_recent_execution"] = most_recent_execution

    st.session_state["results"] = get_results_table()
    st.session_state["fig"] = create_fig_visualization(
        st.session_state["results"])


def generate_configs_file_yaml(col: st.delta_generator.DeltaGenerator) -> None:
    """
    Parse the input data introduced by the user and generate the config file in
    yaml format

    Args:
        col: Defines the column where to place the spinner.
    """
    with col:
        with st.spinner("Reading configs and generating configuration "
                        "file .yaml... ⏳"):

            yaml_file = st.session_state["config_file"]
            # make sure that the order of the steps is correct
            if st.session_state["functionality"] == "Benchmark":
                yaml_file["steps"] = {
                    "data": yaml_file["steps"]["data"],
                    "benchmark": yaml_file["steps"]["benchmark"]
                }
            elif st.session_state["functionality"] == "Train":
                yaml_file["steps"] = {
                    "data": yaml_file["steps"]["data"],
                    "model": yaml_file["steps"]["model"]
                }

            with open(config_file_path, 'w') as file:
                yaml.safe_dump(yaml_file, file,
                               default_flow_style=False,
                               sort_keys=False)
                
            return yaml_file
