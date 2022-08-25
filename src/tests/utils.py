import pandas as pd


def mock_up_read_dataframe() -> pd.DataFrame:
    """
    This method generates a dataframe for testing purposes.

    Returns:
        (pd.DataFrame): fake dataframe
    """
    data = {
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'target1': [10, 20, 30],
        'target2': [40, 50, 60]
    }
    return pd.DataFrame(data)
