import sys

import pandas as pd
from sklearn import model_selection
from .parameters import (
    NUM_SPLITS, TEST_PCT_SPLIT, RANDOM_SEED
)


def generate_splits(
        input_file: str, output_file: str, num_splits: int,
        test_pct: float, seed: int) -> None:
    """
    Split dataset using the options specified. This is done in the following
    order:
    1. Splits indices are generated
    2. For each split, train and test subsets are labelled
    3. A new dataset is generated concatenating and labelling the split
       datasets.
    4. The new dataset containing all the splits is stored.

    Args:
        input_file: Input file.
        output_file: Output file.
        num_splits: Number of splits to generate.
        test_pct: Percentage for each split to be considered test data.
        seed: Seed to use for splitting.
    """
    # Create output path if it does not exist
    data = pd.read_csv(input_file)

    # Generate split indices
    splits = model_selection.KFold(
        n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    df_splits = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(splits.split(data)):

        print(f'Generating split: {i}')
        df_split = data.copy(deep=True)
        df_split['split'] = i
        df_split['train_test'] = 'train'
        df_split.loc[test_index, 'train_test'] = 'test'
        df_splits = pd.concat([df_splits, df_split])

    df_splits.to_csv(output_file, index=None)


if __name__ == '__main__':
    input_file, output_file = sys.argv[1], sys.argv[2]
    generate_splits(input_file, output_file, NUM_SPLITS, TEST_PCT_SPLIT, RANDOM_SEED)
