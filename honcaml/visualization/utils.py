import streamlit as st


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Upload your configuration file .yaml 📄 or set your "
            "configuration parameters manually\n"
            "2. Press the `Run` button and wait until the execution finishes\n"
        )
        st.markdown("---")
        st.markdown(
            "## About\n"
            "HoNCAML(Holistic No Code Automated Machine Learning) is a tool "
            "aimed to run automated machine learning \
            pipelines for problems of different nature; main types of pipeline"
            " would be:\n"
            "1. Training the best possible model for the problem at hand\n"
            "2. Use this model to predict other instances\n\n"

            "At this moment, the following types of problems are supported:\n"
            "- Regression\n"
            "- Classification\n"
        )


def download_logs_button(col):
    with open('logs.txt', 'r') as logs_reader:
        col.download_button(label="Download logs as .txt",
                            data=logs_reader.read(),
                            file_name='logs.txt')


def error_message():
    with open('errors.txt') as errors_reader:
        st.error("**There was an error during the execution:**\n\n" +
                 errors_reader.read(), icon='🚨')


def align_button(col):
    col.write("\n")
    col.write("\n")
