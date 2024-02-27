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


def replace_string_column_to_numeric(ds: pd.Series) -> pd.DataFrame:
    """
    Replace string values by numerical ones, without any specific order.

    Args:
        data: Input dataset.
        target: Target column name.

    Returns:
        Dataset with numeric target.
    """
    label_encoder = LabelEncoder()
    ds = label_encoder.fit_transform(ds)
    return ds
