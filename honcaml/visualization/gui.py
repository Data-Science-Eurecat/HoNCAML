import yaml
import streamlit as st
from manual_configs import manual_configs_elements
import app_execution
import visualization as viz
from utils import (set_current_session,
                   sidebar,
                   change_configs_mode,
                   reset_config_file,
                   upload_data_file,
                   initialize_config_file,
                   write_uploaded_file,
                   define_metrics,
                   config_file_path,
                   error_message,
                   download_logs_button,
                   align_button,
                   download_trained_model_button,
                   create_output_folder)

# from streamlit_ttyd import terminal
# from streamlit.components.v1 import iframe
# from port_for import get_port


def main():
    """Main execution function."""
    st.set_page_config(page_title="HoNCAML", layout="wide")
    st.header("HoNCAML")
    sidebar()

    if "current_session" not in st.session_state:
        st.session_state["current_session"] = set_current_session()

    col1, data_upload_col = st.columns(2)
    configs_mode = col1.radio("Introduce configurations via:",
                              ("Manually", "Config file .yaml",
                               "Paste your configs"),
                              on_change=change_configs_mode)
    col1.write("")

    # place the data preview container before the configs mode selector
    data_preview_container = st.container()

    # if "Manual" option selected, add radio selectors for the level of
    # configurations (basic or advanced) and the functionality (benchmark,
    # train or test
    if configs_mode == "Manually":
        col1_1, col1_2 = col1.columns(2)
        st.session_state["configs_level"] = \
            col1_1.radio("Configurations", ("Basic", "Advanced"),
                         on_change=reset_config_file)
        st.session_state["functionality"] = \
            col1_2.radio("Functionality", ("Benchmark", "Train", "Predict"),
                         on_change=reset_config_file)
        st.write("")

    # if "Config file .yaml" option selected, add file uploader yaml
    elif configs_mode == "Config file .yaml":
        col1.file_uploader(
            "Upload your configurations file .yaml",
            type=[".yaml"],
            key="uploaded_file"
        )

    # if "Paste your configs" option selected, add text input area to paste or
    # write the configs
    elif configs_mode == "Paste your configs":
        st.session_state["text_area"] = \
            yaml.safe_load(st.text_area("Paste here your config file in json "
                                        "or yaml format"))

    # upload data file
    st.session_state['data_uploaded'] = \
        upload_data_file(data_upload_col,
                         data_preview_container,
                         configs_mode)

    # if "Manual" option selected, add manual configuration elements
    if configs_mode == "Manually":
        initialize_config_file()
        manual_configs_elements()

    # add "Run" button to execute the app
    col1, col2 = st.columns([1, 8])
    button = col1.button("Run")

    # when the "Run" button is pressed, execute the app
    if button:
        # check that the datafile is uploaded
        if not st.session_state["data_uploaded"]:
            st.session_state["submit"] = False
            st.warning('You must upload data file', icon="⚠️")
        else:
            st.session_state["submit"] = True

            if configs_mode == "Manually":
                with col2:
                    app_execution.generate_configs_file_yaml()
                    create_output_folder()
                    app_execution.run()

            elif configs_mode == "Config file .yaml":
                if "uploaded_file" in st.session_state:
                    write_uploaded_file(
                        st.session_state["uploaded_file"])
                    define_metrics()
                    with col2:
                        app_execution.run()
                else:
                    st.warning('You must provide a configuration file',
                               icon="⚠️")

            elif configs_mode == "Paste your configs":
                with open(config_file_path, "w") as file:
                    if "text_area" in st.session_state:
                        yaml.safe_dump(st.session_state["text_area"], file,
                                       default_flow_style=False,
                                       sort_keys=False)
                        with col2:
                            app_execution.run()
                    else:
                        st.warning("You must introduce your configurations in "
                                   "the text area",
                                   icon="⚠️")

    if st.session_state.get("submit"):

        if st.session_state["functionality"] == "Benchmark":
            col2_1, col2_2 = col2.columns([1, 4])
            if st.session_state["process_poll"] == 0:
                col2_2.success('Execution successful!', icon="✅")
                app_execution.process_results()

                viz.display_best_hyperparameters()

                col1, col2 = st.columns([1, 8])
                results_display = col1.radio(
                    "Display results as:", ("Table", "BarChart")
                )
                align_button(col2)
                col2.download_button(
                    label="Download results as .csv",
                    data=st.session_state["results"].to_csv().encode('utf-8'),
                    file_name='results.csv')

                viz.display_results(results_display)

            else:
                error_message()

            download_logs_button(col2_1)

        elif st.session_state["functionality"] == "Train":
            if st.session_state["process_poll"] == 0:
                col2.success('Execution successful!', icon="✅")
            else:
                error_message()
            pass
            col1, col2 = st.columns(2)
            download_logs_button(col1)
            download_trained_model_button(col2)

        elif st.session_state["functionality"] == "Test":
            if st.session_state["process_poll"] == 0:
                col2.success('Execution successful!', icon="✅")
            else:
                error_message()
            pass
            download_logs_button()


if __name__ == '__main__':
    main()
