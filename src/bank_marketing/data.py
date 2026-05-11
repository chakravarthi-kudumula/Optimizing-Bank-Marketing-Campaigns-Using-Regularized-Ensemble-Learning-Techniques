from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]


def load_bank_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected columns found: {extra}")


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "y",
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column].map({"yes": 1, "no": 0})
    if y.isna().any():
        raise ValueError("Target column must contain only 'yes' and 'no'.")
    return X, y.astype(int)


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
