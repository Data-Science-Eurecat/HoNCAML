import os
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px
from constants import benchmark_results_path
from honcaml.config.defaults.model_step import default_model_step


def data_previsualization(
        data_preview_container: st.delta_generator.DeltaGenerator) -> None:
    """
    Display a table with a preview of 5 lives of the data file

    Args:
        data_preview_container: Defines the streamlit container where to place
            the table.
    """
    data = st.session_state["data_uploaded"]
    with data_preview_container.expander("Data preview"):
        st.write(data.head())


def get_results_table() -> pd.DataFrame:
    """
    Load and process the results.csv file generated during the benchmark
    execution and process it in order to be posteriorly used to create the
    visualization.

    Returns:
        results: Pandas dataframe with the processed data.
    """
    # find the most recent execution folder
    file_path = os.path.join('../../', benchmark_results_path,
                             st.session_state["current_session"],
                             st.session_state["most_recent_execution"],
                             'results.csv')
    results = pd.read_csv(file_path)

    results['model'] = \
        results['config/model_module'].apply(lambda x: x.split('.')[-1])

    results['configs'] = results.apply(
        lambda x: " ".join(
            list(filter(None, [col.split('/')[-1] + ":" + str(x[col])
                               if not pd.isna(x[col])
                               else ""
                               for col in list(
                    filter(lambda k: 'config/param_space' in k,
                           results.columns))]))
        ), axis=1
    )

    results['model_configs'] = \
        results['model'] + '<br>' + \
        results['configs'].replace(r' ', '<br>', regex=True)

    if st.session_state["configs_level"] == "Advanced":
        st.session_state["benchmark_metrics"] = \
            st.session_state["config_file"]["steps"]["benchmark"]["transform"][
                "metrics"]
    elif st.session_state["configs_level"] == "Basic":
        st.session_state["benchmark_metrics"] = \
            list(set(st.session_state["problem_type_metrics"])
                 .intersection(set(results.columns)))
                
    b_met = st.session_state["benchmark_metrics"]
    benchmark_metrics = b_met if isinstance(b_met, list) else [b_met]
    cols_list = ['model', 'configs', 'model_configs'] + benchmark_metrics       
    results = results[cols_list]

    results = results.drop_duplicates(subset=['model', 'configs']) \
        .reset_index() \
        .drop(columns=['index'])
    return results

def create_fig_visualization(results) -> object:
    """
    Creates a comparison visualization of the trained models and their metrics.

    Args:
        results: Pandas dataframe containing the models and metrics values.

    Returns:
        fig: Plotly figure.
    """
    height = int(len(results.index) / 3 + 3)

    results_melted = \
        results[['model_configs'] + st.session_state["benchmark_metrics"]] \
        .melt(
              id_vars=['model_configs'],
              value_vars=st.session_state["benchmark_metrics"],
              var_name='metric'
        )

    fig = px.bar(
        results_melted,
        x="value",
        y="model_configs",
        color="model_configs",
        facet_col="metric",
        facet_col_wrap=3,
        facet_row_spacing=0.75 / height,
        height=height * 100
    )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(title=None, matches=None, showticklabels=True)
    fig.update_layout(legend=dict(title='Model & Configs'))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig


def display_best_hyperparameters() -> None:
    """
    Display best model and hyperparameters after running the benchmark.
    """
    yaml_file_path = \
        os.path.join('../../', benchmark_results_path,
                     st.session_state["current_session"],
                     st.session_state["most_recent_execution"],
                     'best_config_params.yaml')
    with open(yaml_file_path, 'r') as stream:
        config_file = yaml.safe_load(stream)
    hyperparams = ""
    for elem, value in config_file['params'].items():
        hyperparams = hyperparams + '**' + elem + '**: **' + str(value) + \
                      '**, '
    hyperparams = hyperparams[:len(hyperparams) - 2]

    st.markdown(
        f"The best model is **{config_file['module']}**\n\n"
        f"And the best set of hyperparameter are: "
        f"**{hyperparams}**"
    )


def display_results(results_display: str) -> None:
    """
    Display results in the form of table or barchart.

    Args:
        results_display: ["Table", "BarChart"] Define the format of the results
            display
    """
    if results_display == "Table":
        results_table = st.session_state["results"] \
            .set_index(['model', 'configs']) \
            .drop(columns=['model_configs'])
        results_table.columns = results_table.columns.str.replace('_', ' ')
        st.table(results_table)

    elif results_display == "BarChart":
        _, col2, _ = st.columns([1, 5, 1])
        col2.plotly_chart(st.session_state["fig"],
                          use_container_width=False)

    st.write(f'Execution ID: {st.session_state["most_recent_execution"]}')


def display_results_train() -> None:
    """
    Display name of the trained model and the hyperparameters used.
    """
    model_configs = st.session_state["config_file"]["steps"]["model"]
    # get model and params specified in the config file
    if model_configs["transform"].get("fit"):
        model = model_configs["transform"]["fit"]["estimator"]["module"]
        params = model_configs["transform"]["fit"]["estimator"]["params"]
    # get model and params specified by default
    else:
        problem_type = st.session_state["config_file"]["global"][
            "problem_type"]
        model = default_model_step["transform"]["fit"]["estimator"][
            problem_type]["module"]
        params = default_model_step["transform"]["fit"]["estimator"][
            problem_type]["params"]
    st.write(f"Model: **{model}**")
    st.write(f"Parameters: **{params}**")
