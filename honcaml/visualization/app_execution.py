import os
import streamlit as st
from subprocess import Popen
from visualization import get_results_table, create_fig_visualization


def run(uploaded_file, col2):
    if "submit" not in st.session_state:
        st.session_state["submit"] = True
    with open("config_file.yaml", "w") as f:
        f.write(uploaded_file.getvalue().decode("utf-8"))
        f.close()
    with col2:
        with st.spinner("Running... This may take a while⏳"):
            log = open('logs.txt', 'w')
            err = open('errors.txt', 'w')
            # port = get_port((5000, 7000))
            # process = Popen(f'ttyd --port {port} --once honcaml -c config_file.yaml', shell=True)
            process = Popen(f'cd ../.. && honcaml -c config_file.yaml',
                            shell=True,
                            stdout=log,
                            stderr=err)
            # process = Popen(f'ls', shell=True, stdout=log, stderr=err)
            # host = "http://localhost"
            # iframe(f"{host}:{port}", height=400)
            process.wait()
            process_poll = process.poll()
            st.session_state["process_poll"] = process_poll

    most_recent_execution = max(os.listdir('../../honcaml_reports'))
    st.session_state["most_recent_execution"] = most_recent_execution

    results = get_results_table(most_recent_execution)
    st.session_state["results"] = results
    fig = create_fig_visualization(results, most_recent_execution)
    st.session_state["fig"] = fig


def generate_configs_file_yaml():
    pass