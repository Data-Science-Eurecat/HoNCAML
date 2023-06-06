import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


def get_results_table(most_recent_execution):
    """
    Load and process the results.csv file generated during the benchmark
    execution and process it in order to be posteriorly used to create the
    visualization

    Args:
        most_recent_execution: date and time of the most recent execution

    Returns:
        results: pandas dataframe with the processed data
    """
    # find the most recent execution folder
    file_path = f'../../honcaml_reports/{most_recent_execution}/results.csv'
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

    results = results[
        ['model', 'configs', 'model_configs'] + st.session_state["metrics"]
        ]

    results = results.drop_duplicates(subset=['model', 'configs']) \
        .reset_index() \
        .drop(columns=['index'])
    return results


def create_fig_visualization(results):
    """
    Creates a comparison visualization of the trained models and their metrics

    Args:
        results: pandas dataframe containing the models and metrics values

    Returns:
        fig: plotly figure
    """
    results_melted = results[['model_configs'] + st.session_state["metrics"]] \
        .melt(
        id_vars=['model_configs'],
        value_vars=st.session_state["metrics"],
        var_name='metric'
    )

    height = int(sum(results_melted['metric'] == 'mean_squared_error') / 3 + 3)
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


def display_best_hyperparameters():
    """
    Display best model and hyperparameters after the benchmark
    """
    yaml_file_path = \
        f'../../honcaml_reports/{st.session_state["most_recent_execution"]}' \
        f'/best_config_params.yaml'
    with open(yaml_file_path, 'r') as stream:
        config_file = yaml.safe_load(stream)
    hyperparams = ""
    for elem, value in config_file['hyper_parameters'].items():
        hyperparams = hyperparams + '**' + elem + '**: **' + str(value) + \
                      '**, '
    hyperparams = hyperparams[:len(hyperparams) - 2]

    st.markdown(
        f"The best model is **{config_file['module']}**\n\n"
        f"And the best set of hyperparameter are: "
        f"**{hyperparams}**"
    )


def display_results(results_display):
    """
    Display results in the form of table or barchart
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
