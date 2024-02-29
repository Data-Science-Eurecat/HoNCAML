import pandas as pd
from sklearn.preprocessing import LabelEncoder


def remove_non_numerical_features(
        data: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Remove non-numerical features from dataset.

    Args:
        data: Input dataset.
        target: Target column name.

    Returns:
        Dataset without non-numerical features.
    """
    remove = []
    for col in data.columns:
        if target != col:
            if data[col].dtype == 'object':
                remove.append(col)
    data = data.drop(columns=remove)
    return data


def replace_string_columns_to_numeric(
        data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Replace string columns by numerical ones by label encoding them,
    using the same encoding logic.

    Args:
        data: Input dataset.
        cols: Column/s to convert to numeric.

    Returns:
        Dataset with numeric target.
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(data[cols[0]])
    for col in cols:
        data[col] = label_encoder.transform(data[col])
    return data
