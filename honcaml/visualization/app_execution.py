import os
import streamlit as st
from subprocess import Popen
from visualization import get_results_table, create_fig_visualization


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
        if st.session_state["configs_level"] == "Basic":
            process = Popen(f'cd ../.. && honcaml -b config_file.yaml', shell=True)
                            # , stdout=log, stderr=err)
        else:
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
    pass
