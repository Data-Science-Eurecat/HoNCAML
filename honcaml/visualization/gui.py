import streamlit as st
from manual_configs import manual_configs_elements
from app_execution import (generate_configs_file_yaml,
                           run,
                           process_results)
from visualization import (display_best_hyperparameters,
                           display_results_train,
                           display_results)
from utils import (set_current_session,
                   sidebar,
                   define_metrics,
                   define_functionality_configs_level,
                   error_message,
                   create_output_folder,
                   create_logs_folder,
                   warning)
from define_config_file import initialize_config_file
from extract import add_init_input_elements
from load import (load_uploaded_file,
                  load_text_area_configs,
                  download_logs_button,
                  download_trained_model_button,
                  download_predictions_button,
                  download_benchmark_results_button)
import yaml

# from streamlit_ttyd import terminal
# from streamlit.components.v1 import iframe
# from port_for import get_port


def main():
    """Main execution function."""

    # set page configs
    st.set_page_config(page_title="HoNCAML", layout="wide")
    st.header("HoNCAML")
    sidebar()

    if "current_session" not in st.session_state:
        st.session_state["current_session"] = set_current_session()

    # add initial input elements: configs mode, config file uploader,
    # data uploader, target selector, configs level, functionality, and data
    # pre-visualization
    add_init_input_elements()

    configs_mode = st.session_state["configs_mode"]

    # if "Manual" option selected, add manual configuration elements
    if configs_mode == "Manually":
        initialize_config_file()
        manual_configs_elements()

    # add "Run" button to execute the app
    col1, col2 = st.columns([1, 8])
    run_button = col1.button("Run")

    # add a button to download the config file
    yaml_file = generate_configs_file_yaml(col2)
    yaml_file_string = yaml.safe_dump(yaml_file, default_flow_style=False,
                                      sort_keys=False)
    col1.download_button("Download config file", data=yaml_file_string,
                         file_name="config_file.yaml")

    # when the "Run" button is pressed, execute the app
    if run_button:
        # check that the datafile is uploaded
        if st.session_state.get("data_uploaded") is not None:
            st.session_state["submit"] = True
            create_logs_folder()

            if configs_mode == "Manually":
                create_output_folder()
                run(col2)

            elif configs_mode == "Config file .yaml":
                if "uploaded_file" in st.session_state:
                    load_uploaded_file()
                    define_metrics()
                    define_functionality_configs_level()
                    run(col2)
                else:
                    warning("config_file")

            elif configs_mode == "Paste your configs":
                if "text_area" in st.session_state:
                    load_text_area_configs()
                    define_functionality_configs_level()
                    run(col2)
                else:
                    warning("text_area")

        else:
            st.session_state["submit"] = False
            warning("data_file")

    if st.session_state.get("submit"):

        if st.session_state["process_poll"] == 0:
            col2.success('Execution successful!', icon="✅")
        else:
            error_message()

        if st.session_state["functionality"] == "Benchmark":
            download_logs_button()
            if st.session_state["process_poll"] == 0:
                process_results()
                display_best_hyperparameters()

                col1, col2 = st.columns([1, 5])
                results_display = col1.radio(
                    "Display results as:", ("Table", "BarChart")
                )
                col2.write("\n")
                col2.write("\n")

                download_benchmark_results_button(col2)
                display_results(results_display)

        elif st.session_state["functionality"] == "Train":
            col1, col2 = st.columns(2)
            display_results_train()
            download_logs_button(col1) 
            download_trained_model_button()

        elif st.session_state["functionality"] == "Predict":
            col1, col2 = st.columns([1, 3])
            download_predictions_button(col1)
            download_logs_button(col2)


if __name__ == '__main__':
    main()
