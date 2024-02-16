import os
import shutil
import yaml
import streamlit as st
from subprocess import Popen
from constants import benchmark_results_path, config_file_path, logs_path
from honcaml.visualization.visualization import (
    get_results_table, create_fig_visualization)


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
            log = open(os.path.join(logs_path, 'logs.txt'), 'w')
            err = open(os.path.join(logs_path, 'errors.txt'), 'w')
            process = Popen(f'honcaml -c {config_file_path}',
                            shell=True, stdout=log, stderr=err)
            process.wait()
            process_poll = process.poll()
            st.session_state["process_poll"] = process_poll


def process_results() -> None:
    """
    Put results directly within streamlit reports, instead of having an
    intermediate ID.
    Create a table and a figure with the results of the execution
    """
    internal_id = os.listdir(benchmark_results_path)[0]
    internal_path = os.path.join(benchmark_results_path, internal_id)
    objects_to_copy = os.listdir(internal_path)
    for object_ in objects_to_copy:
        object_path = os.path.join(internal_path, object_)
        new_object_path = os.path.join(benchmark_results_path, object_)
        if os.path.isfile(object_path):
            shutil.copy(object_path, new_object_path)
        elif os.path.isdir(object_path):
            shutil.copytree(object_path, new_object_path)
    shutil.rmtree(internal_path)

    st.session_state["results"] = get_results_table()
    st.session_state["fig"] = \
        create_fig_visualization(st.session_state["results"])


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
            elif st.session_state["functionality"] in ["Train", "Predict"]:
                yaml_file["steps"] = {
                    "data": yaml_file["steps"]["data"],
                    "model": yaml_file["steps"]["model"]
                }

            with open(config_file_path, 'w') as file:
                yaml.safe_dump(yaml_file, file,
                               default_flow_style=False,
                               sort_keys=False)

            return yaml.safe_dump(yaml_file, default_flow_style=False,
                                  sort_keys=False)
