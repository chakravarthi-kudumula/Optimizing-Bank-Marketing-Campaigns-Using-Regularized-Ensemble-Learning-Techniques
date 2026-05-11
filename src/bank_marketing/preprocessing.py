from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_marketing.features import (
    FeatureEngineer,
    categorical_columns,
    numeric_columns,
)


def build_preprocessor(drop_duration: bool = True, scale_numeric: bool = True) -> Pipeline:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns(drop_duration)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns(),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("features", FeatureEngineer(drop_duration=drop_duration)),
            ("columns", column_transformer),
        ]
    )


def transformed_dataframe(
    preprocessor: Pipeline,
    X: pd.DataFrame,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    values = preprocessor.transform(X)
    feature_names = preprocessor.named_steps["columns"].get_feature_names_out()
    return pd.DataFrame(values, columns=feature_names, index=index)
