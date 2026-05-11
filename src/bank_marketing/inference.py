from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def load_model(path: str | Path):
    return joblib.load(path)


def predict_dataframe(model, X: pd.DataFrame) -> pd.DataFrame:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)
    output = X.copy()
    output["subscription_probability"] = probabilities
    output["subscription_prediction"] = predictions
    output["subscription_prediction_label"] = output["subscription_prediction"].map(
        {1: "yes", 0: "no"}
    )
    return output
