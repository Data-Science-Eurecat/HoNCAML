import sys
import time

import pandas as pd

from src import utils
from src.parameters import BENCHMARK_OPTIONS, DATASET_OPTIONS


def run_framework(
        framework: str, input_file: str, output_file: str,
        dataset_options: str, benchmark_options: dict) -> None:
    """
    Execute framework to get benchmark results for dataset. This is done by:
    1. Detect problem type related to dataset (regression, classification)
    2. Read all seeds and splits. For each combination of seed and split, do
       the following:
      - Preprocess data if applies for the specified framework
      - Run the framework over the training set to obtain the best model
      - Use the test set and the trained model to predict values
      - Store dataset with predictions and ground truths in the specified path

    Args;
        framework: Framework to consider.
        input_file: Input file.
        output_file: Output file.
        datasets_options: Datasets parameters.
        benchmark_options: General options for benchmark.
    """
    # Set dataset options
    dataset = input_file.split('/')[-1].split('.')[0]
    problem_type = dataset_options[dataset]['type']
    target = dataset_options[dataset]['target']

    # Initialize automl class
    automl_class = utils.retrieve_problem_class(
        framework, problem_type)
    data = pd.read_csv(input_file)
    seeds = list(data['seed'].unique())
    splits = list(data['split'].unique())

    df_predictions = pd.DataFrame()

    for seed in seeds:

        print(f'Using seed: {seed}')
        seed_data = data.loc[data['seed'] == seed].copy(deep=True)

        for split in splits:

            print(f'Using split: {split}')
            split_data = seed_data.loc[
                seed_data['split'] == split].copy(deep=True)
            split_data = split_data.drop(columns='split').reset_index(
                drop=True)

            # Process data
            train_idx = split_data['train_test'] == 'train'
            test_idx = split_data['train_test'] == 'test'

            split_data = split_data.drop(columns='train_test')
            split_data = automl_class.preprocess_data(
                split_data, target, dataset)

            # Get training data
            df_train = split_data.loc[train_idx].copy(
                deep=True).reset_index(drop=True)

            # Search for best model
            benchmark_options['dataset'] = dataset
            benchmark_time = time.time()
            automl_class.search_best_model(df_train, target, benchmark_options)
            benchmark_time = time.time() - benchmark_time

            # Predict with test set
            df_prediction = split_data.loc[test_idx].copy(
                deep=True).reset_index(drop=True)
            predict_time = time.time()
            try:
                df_prediction['y_pred'] = automl_class.predict(
                    df_prediction, target)
            except ValueError:
                print('Error during prediction; passing')
                df_prediction['y_pred'] = None
            predict_time = time.time() - predict_time

            # Format output dataset
            df_prediction['split'] = split
            df_prediction['benchmark_time'] = benchmark_time
            df_prediction['predict_time'] = predict_time
            df_prediction = df_prediction.rename(columns={target: 'y_true'})

            # Append to predictions
            df_prediction['seed'] = seed
            df_predictions = pd.concat([df_predictions, df_prediction])

        cols_output = [
            'seed', 'split', 'benchmark_time', 'predict_time',
            'y_true', 'y_pred']
        df_predictions = df_predictions[cols_output]

    # Store file
    df_predictions.to_csv(output_file, index=None)


if __name__ == '__main__':
    framework, input_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    run_framework(
        framework, input_file, output_file, DATASET_OPTIONS, BENCHMARK_OPTIONS)
