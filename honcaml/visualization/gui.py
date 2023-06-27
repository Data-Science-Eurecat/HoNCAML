import yaml
import streamlit as st
from manual_configs import manual_configs_elements
from app_execution import (generate_configs_file_yaml,
                           run,
                           process_results)
from visualization import (data_previsualization,
                           display_best_hyperparameters,
                           display_results_train,
                           display_results)
from constants import config_file_path
from utils import (set_current_session,
                   sidebar,
                   define_metrics,
                   error_message,
                   create_output_folder,
                   warning)
from define_config_file import (initialize_config_file)
from extract import (extract_configs_mode,
                     extract_configs_level,
                     extract_functionality,
                     extract_configs_file_yaml,
                     extract_configs_file_text_area,
                     extract_data_file,
                     extract_target)
from load import (load_data_file,
                  load_uploaded_file,
                  download_logs_button,
                  download_trained_model_button,
                  download_predictions_button,
                  download_benchmark_results_button)

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

    col1, data_upload_col = st.columns(2)

    # define configs mode: Manually, Config file .yaml, or Paste your configs
    configs_mode = extract_configs_mode(col1)
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
        load_data_file(st.session_state["data_uploaded"])
        # add target selector
        extract_target(data_upload_col, configs_mode)
        # pre-visualize data
        data_previsualization(data_preview_container)

    # if "Manual" option selected, add manual configuration elements
    if configs_mode == "Manually":
        initialize_config_file()
        manual_configs_elements()

        # add a preview of the config file
        with st.expander("Config file preview"):
            st.write(st.session_state)

    # add "Run" button to execute the app
    col1, col2 = st.columns([1, 8])
    button = col1.button("Run")

    # when the "Run" button is pressed, execute the app
    if button:
        # check that the datafile is uploaded
        if st.session_state.get("data_uploaded") is not None:
            st.session_state["submit"] = True

            if configs_mode == "Manually":
                with col2:
                    generate_configs_file_yaml()
                    create_output_folder()
                    run()

            elif configs_mode == "Config file .yaml":
                if "uploaded_file" in st.session_state:
                    load_uploaded_file(st.session_state["uploaded_file"])
                    define_metrics()
                    with col2:
                        run()
                else:
                    warning("config_file")

            elif configs_mode == "Paste your configs":
                with open(config_file_path, "w") as file:
                    if "text_area" in st.session_state:
                        yaml.safe_dump(st.session_state["text_area"], file,
                                       default_flow_style=False,
                                       sort_keys=False)
                        with col2:
                            run()
                    else:
                        warning("text_area")

        else:
            st.session_state["submit"] = False
            warning("data_file")

    if st.session_state.get("submit"):

        if st.session_state["functionality"] == "Benchmark":
            col2_1, col2_2 = col2.columns([1, 4])
            if st.session_state["process_poll"] == 0:
                col2_2.success('Execution successful!', icon="✅")
                process_results()

                display_best_hyperparameters()

                col1, col2 = st.columns([1, 8])
                results_display = col1.radio(
                    "Display results as:", ("Table", "BarChart")
                )
                col2.write("\n")
                col2.write("\n")
                download_benchmark_results_button(col2)

                display_results(results_display)

            else:
                error_message()

            download_logs_button(col2_1)

        elif st.session_state["functionality"] == "Train":
            if st.session_state["process_poll"] == 0:
                col2.success('Execution successful!', icon="✅")
            else:
                error_message()

            col1, col2 = st.columns(2)
            display_results_train()
            download_logs_button(col1)
            download_trained_model_button()

        elif st.session_state["functionality"] == "Predict":
            if st.session_state["process_poll"] == 0:
                col2.success('Execution successful!', icon="✅")
            else:
                error_message()
            col1, col2 = st.columns([1, 3])
            download_predictions_button(col1)
            download_logs_button(col2)


if __name__ == '__main__':
    main()
