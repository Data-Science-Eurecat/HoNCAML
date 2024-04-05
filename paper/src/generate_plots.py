import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.parameters import DATASET_OPTIONS, PLOT_OPTIONS


def generate_plots(input_path: str, output_path: str,
                   dataset_options: dict, parameters: dict):
    """
    Generate plots and final summaries from benchmark results.
    First, all result files are read and stored in a single dataset.
    Then, summary tables with rankings and variability are stored for all
    frameworks and datasets.
    It also generates a time plot, to compare both benchmark time and predict
    time for all frameworks and datasets.
    Finally, it generates, for each type of problem, a plot with as many frames
    as datasets considered. In each frame, the results are shown in a scatter
    plot, with axes corresponding to metric (Y) and time (X). Each observation
    correspond of the mean of the seeds results for each framework.

    Args:
        input_path: Input path where results are stored.
        output_path: Output path where plots are stored.
        datasets_options: Datasets parameters.
        parameters: Plot parameters.
    """
    framework_files = [x for x in os.listdir(input_path) if '.csv' in x]

    df_results = pd.DataFrame()
    for framework_file in framework_files:
        framework = framework_file.split('.')[0]
        filepath = os.path.join(input_path, framework_file)
        df_framework = pd.read_csv(filepath)
        df_framework['framework'] = framework
        df_results = pd.concat([df_results, df_framework])

    # Get all datasets for a specific type
    problem_types = {'regression': [], 'classification': []}
    for dataset in list(dataset_options.keys()):
        dataset_problem_type = dataset_options[dataset]['type']
        problem_types[dataset_problem_type].append(dataset)

    print('Generating summary tables')
    for problem_type in list(problem_types.keys()):
        rank_filepath = os.path.join(
            output_path, 'rank_' + problem_type + '.csv')
        var_filepath = os.path.join(
            output_path, 'var_' + problem_type + '.csv')
        df_rank = pd.DataFrame(columns=['framework', 'rank'])
        df_var = pd.DataFrame(columns=['dataset', 'framework', 'std'])
        metric = parameters['metrics'][problem_type]
        minimize = parameters['minimize'][problem_type]
        datasets = problem_types[problem_type]
        for dataset in datasets:
            df_dataset = df_results.loc[
                df_results['dataset'] == dataset].reset_index(
                    drop=True).copy(deep=True)
            # Generate ranks
            df_rank_dataset = df_dataset.sort_values(
                metric, ascending=minimize)[['dataset', 'framework']]
            df_rank_dataset['rank'] = range(len(df_rank_dataset))
            df_rank_dataset['rank'] = df_rank_dataset['rank'] + 1
            df_rank = pd.concat([df_rank, df_rank_dataset])
            # Generate variabilities
            df_var_dataset = df_dataset.groupby(
                ['dataset', 'framework'])[metric].apply(
                    np.std).reset_index().rename(columns={metric: 'std'})
            df_var = pd.concat([df_var, df_var_dataset])
        df_rank.to_csv(rank_filepath, index=None)
        df_var.to_csv(var_filepath, index=None)

    frameworks = list(df_results['framework'].unique())
    print('Generating time plot')
    for framework in frameworks:
        df_framework = df_results.loc[
            df_results['framework'] == framework].copy(deep=True)
        plt.scatter(df_framework['benchmark_time'],
                    df_framework['predict_time'],
                    label=framework,
                    c=parameters['colors'][framework],
                    marker=parameters['markers'][framework])
    store_filepath = os.path.join(output_path, 'time.png')
    plt.legend(loc='upper center')
    plt.savefig(store_filepath)
    plt.clf()

    print('Generating metrics plots')
    n_cols = parameters['n_cols']
    for problem_type in list(problem_types.keys()):
        store_filepath = os.path.join(output_path, problem_type + '.png')
        metric = parameters['metrics'][problem_type]
        datasets = problem_types[problem_type]
        n_rows = (len(datasets) / n_cols).__ceil__()
        i = 1
        fig, axes = plt.subplots(n_rows, n_cols)
        for dataset in datasets:
            ax = axes[(i - 1) // n_cols, (i - 1) % n_cols]
            df_dataset = df_results.loc[
                df_results['dataset'] == dataset].reset_index(
                    drop=True).copy(deep=True)
            xmax = max(df_dataset[parameters['x_axis']]) * parameters[
                'ratio_max_range']
            ymax = max(df_dataset[metric]) * parameters[
                'ratio_max_range']
            df_dataset['color'] = df_dataset['framework'].map(
                parameters['colors'])
            frameworks = list(df_dataset['framework'].unique())
            for framework in frameworks:
                df_framework = df_dataset.loc[
                    df_dataset['framework'] == framework]
                df_framework = df_framework.groupby(
                    ['dataset', 'framework'])[
                        [metric, parameters['x_axis']]].mean().reset_index()
                ax.scatter(
                    df_framework[parameters['x_axis']],
                    df_framework[metric], c=parameters['colors'][framework],
                    marker=parameters['markers'][framework])
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_xticks([0, 250, 500, 750, 1000])
            ax.set_title(dataset, fontsize=12)
            ax.hlines(df_dataset.loc[
                df_dataset['framework'] == 'honcaml'][metric].mean(),
                      xmin=0, xmax=xmax, linestyles='dashed',
                      colors=parameters['colors']['honcaml'])
            ax.vlines(df_dataset.loc[
                df_dataset['framework'] == 'honcaml'][
                    parameters['x_axis']].mean(),
                ymin=0, ymax=ymax, linestyles='dashed',
                colors=parameters['colors']['honcaml'])
            i += 1
        pending_indices = list(range(i - 1, n_rows * n_cols))
        for ind in pending_indices:
            ax = axes[(i - 1) // n_cols, (i - 1) % n_cols]
            fig.delaxes(ax=ax)
            i += 1
        plt.tight_layout()
        plt.savefig(store_filepath)
        plt.clf()


if __name__ == '__main__':
    input_path, output_path = sys.argv[1], sys.argv[2]
    generate_plots(input_path, output_path, DATASET_OPTIONS, PLOT_OPTIONS)
