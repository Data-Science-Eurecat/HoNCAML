import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st


def get_results_table(most_recent_execution):
    # find the most recent execution folder
    file_path = f'../../honcaml_reports/{most_recent_execution}/results.csv'
    results = pd.read_csv(file_path)

    results['model'] = \
        results['config/model_module'].apply(lambda x: x.split('.')[-1])

    results['configs'] = results.apply(
        lambda x: 'max_features:' + str(x['config/param_space/max_features']) +
                  ' n_estimators:' +
                  str(int(x['config/param_space/n_estimators']))
        if x['model'] == 'RandomForestRegressor'
        else 'fit_intercept:' + str(x['config/param_space/fit_intercept']),
        axis=1
    )

    results['model_configs'] = results.apply(
        lambda x: x['model'] + '<br>' + x['configs'].split(' ')[0] + '<br>' +
                  x['configs'].split(' ')[1]
        if x['model'] == 'RandomForestRegressor'
        else x['model'] + '<br>' + x['configs'], axis=1
    )

    results = results[
        ['model', 'configs', 'model_configs'] + st.session_state["metrics"]
    ]

    results = results.drop_duplicates(subset=['model', 'configs']) \
        .reset_index() \
        .drop(columns=['index'])
    return results


def create_fig_visualization(results, most_recent_execution):
    # plot a comparison visualization of the trained models and their metrics

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
