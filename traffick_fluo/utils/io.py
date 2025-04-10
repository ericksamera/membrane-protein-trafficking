# traffick_fluo/utils/io.py

import os
import pandas as pd

from typing import Optional


def reorder_feature_columns(df: pd.DataFrame, score_col: Optional[str] = None) -> pd.DataFrame:
    """
    Reorder columns for consistent features.csv layout.

    Order:
        label, cell_id, {score_col}, image, <sorted feature columns...>

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    score_col : str, optional
        Name of the score column (e.g. 'membrane_score').

    Returns
    -------
    pd.DataFrame
        Reordered dataframe.
    """
    fixed = ["label", "cell_id", "image"]
    if score_col:
        fixed.insert(2, score_col)

    fixed = [col for col in fixed if col in df.columns]
    rest = sorted([col for col in df.columns if col not in fixed])
    return df[fixed + rest]


def load_features(path: str) -> pd.DataFrame:
    """
    Load a features CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file containing features.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature values and associated metadata.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")
    return pd.read_csv(path)


def save_features(df: pd.DataFrame, path: str) -> None:
    """
    Save a pandas DataFrame of features to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature data.
    path : str
        Destination path for the output CSV file.
    """
    df.to_csv(path, index=False)
