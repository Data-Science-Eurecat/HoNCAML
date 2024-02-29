import os
import sys

import numpy as np
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
    2. Read all splits. For each split, do the following:
      - Preprocess data if applies for the specified framework
      - Run the framework over the training set to obtain the best model
      - Use the test set and the trained model to predict values
      - Store dataset with predictions and ground truths in the specified path

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

        splits = list(df['split'].unique())

        for metric, args in parameters[problem_type]:

            print(f'Computing metric: {metric}')
            metric_func = getattr(sk_metrics, metric)

            # Compute values for metric
            metric_values = []
            for split in splits:
                df_split = df.loc[df['split'] == split].copy(deep=True)
                metric_value = metric_func(
                    df_split['y_true'], df_split['y_pred'], **args)
                metric_values.append(metric_value)

            metric_values = np.array(metric_values)
            mean_value = np.mean(metric_values)
            std_value = np.std(metric_values)

            # Append valuues to result
            mean_result = (dataset, metric, 'mean', mean_value)
            variance_result = (dataset, metric, 'variance', std_value)
            results.append(mean_result)
            results.append(variance_result)

    # Format results dataset
    df_results = pd.DataFrame(
        results, columns=['dataset', 'metric', 'metric_type', 'value'])

    # Store results dataset
    df_results.to_csv(output_file, index=None)


if __name__ == '__main__':
    input_path, output_file = sys.argv[1], sys.argv[2]
    compute_metrics(input_path, output_file, DATASET_OPTIONS, METRICS)
