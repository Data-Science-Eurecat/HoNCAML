import os
import sys

import pandas as pd
from sklearn import metrics as sk_metrics

from src import processing
from src.parameters import DATASET_OPTIONS, METRICS


def compute_metrics(
        input_path: str, output_file: str, dataset_options: dict,
        parameters: dict) -> None:
    """
    Execute framework to get benchmark results for dataset. This is done by:
    1. Detect problem type related to dataset (regression, classification)
    2. For each seed, compute mean and standard deviation metrics throughout
       all splits
    3. Aggregate values to get global metrics for the framework

    Args;
        input_path: Input path where predictions are stored.
        output_file: Output file.
        datasets_options: Datasets parameters.
        parameters: Metrics parameters.
    """
    datasets = [x.split('.')[0] for x in os.listdir(input_path)]

    results = []
    for dataset in datasets:

        print(f'Handling dataset: {dataset}')
        problem_type = dataset_options[dataset]['type']
        filepath = os.path.join(input_path, dataset + '.csv')
        df = pd.read_csv(filepath)

        if problem_type == 'classification' and df['y_true'].dtype == 'object':
            df = processing.replace_string_columns_to_numeric(
                df, ['y_true', 'y_pred'])

        seeds = list(df['seed'].unique())
        splits = list(df['split'].unique())

        for seed in seeds:

            print(f'Considering seed: {seed}')

            df_seed = df.loc[df['seed'] == seed].copy(deep=True)
            seed_results = {'dataset': dataset, 'seed': seed}
            seed_results['benchmark_time'] = df_seed[
                'benchmark_time'].mean().round(3)
            seed_results['predict_time'] = df_seed[
                'predict_time'].mean().round(3)

            seed_metrics_results = []
            for split in splits:

                print(f'Considering split: {split}')
                df_split = df_seed.loc[df_seed['split'] == split].loc[
                    ~df_seed['y_pred'].isna()].copy(
                    deep=True)

                if len(df_split) == 0:
                    print('All NAS in split')
                    continue

                split_results = {}
                for metric, args in parameters[problem_type]:

                    metric_func = getattr(sk_metrics, metric)
                    metric_value = metric_func(
                        df_split['y_true'], df_split['y_pred'], **args)
                    split_results[metric] = metric_value

                    seed_metrics_results.append(split_results)

            seed_metrics_mean_results = pd.DataFrame(
                seed_metrics_results).mean().round(3).to_dict()
            seed_results = seed_results | seed_metrics_mean_results

            results.append(seed_results)

    # Format results dataset
    df_results = pd.DataFrame(results)

    # Store results dataset
    df_results.to_csv(output_file, index=None)


if __name__ == '__main__':
    input_path, output_file = sys.argv[1], sys.argv[2]
    compute_metrics(input_path, output_file, DATASET_OPTIONS, METRICS)
