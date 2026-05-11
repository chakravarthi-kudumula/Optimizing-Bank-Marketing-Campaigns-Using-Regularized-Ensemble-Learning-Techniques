from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BINARY_COLUMNS = ["default", "housing", "loan"]
NOMINAL_COLUMNS = ["job", "marital", "contact", "poutcome"]
COUNT_COLUMNS = ["campaign", "pdays", "previous"]
EDUCATION_ORDER = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create stable model features from the raw bank marketing columns."""

    def __init__(self, drop_duration: bool = True):
        self.drop_duration = drop_duration
        self.count_caps_: dict[str, float] = {}
        self.balance_min_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        self.count_caps_ = {
            column: float(X[column].quantile(0.99)) for column in COUNT_COLUMNS
        }
        self.balance_min_ = float(X["balance"].min())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = X.copy()

        for column in BINARY_COLUMNS:
            transformed[column] = transformed[column].map({"yes": 1, "no": 0}).astype(int)

        transformed["education_encoded"] = (
            transformed["education"].map(EDUCATION_ORDER).fillna(0).astype(int)
        )
        balance_shifted = (transformed["balance"] - self.balance_min_ + 1).clip(lower=0)
        transformed["balance_log"] = np.log1p(balance_shifted)

        if not self.drop_duration:
            transformed["duration_log"] = np.log1p(transformed["duration"] + 1)

        for column, cap in self.count_caps_.items():
            transformed[column] = transformed[column].clip(upper=cap)

        columns_to_drop = ["education", "month"]
        if self.drop_duration:
            columns_to_drop.append("duration")

        return transformed.drop(columns=columns_to_drop)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            output_features = [
                "age",
                "job",
                "marital",
                "default",
                "balance",
                "housing",
                "loan",
                "contact",
                "day",
            ]
            if not self.drop_duration:
                output_features.append("duration")
            output_features.extend([
                "campaign",
                "pdays",
                "previous",
                "poutcome",
                "education_encoded",
                "balance_log",
            ])
            if not self.drop_duration:
                output_features.append("duration_log")
            return np.array(output_features, dtype=object)

        output_features = list(input_features)
        output_features.append("education_encoded")
        output_features.append("balance_log")
        if not self.drop_duration:
            output_features.append("duration_log")

        for column in ["education", "month"]:
            if column in output_features:
                output_features.remove(column)
        if self.drop_duration and "duration" in output_features:
            output_features.remove("duration")

        return np.array(output_features, dtype=object)


def numeric_columns(drop_duration: bool = True) -> list[str]:
    columns = [
        "age",
        "balance",
        "day",
        "campaign",
        "pdays",
        "previous",
        "default",
        "housing",
        "loan",
        "education_encoded",
        "balance_log",
    ]
    if not drop_duration:
        columns.extend(["duration", "duration_log"])
    return columns


def categorical_columns() -> list[str]:
    return NOMINAL_COLUMNS.copy()
