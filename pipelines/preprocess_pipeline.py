from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bank_marketing.config import load_config, resolve_path
from bank_marketing.data import (
    load_bank_data,
    make_train_test_split,
    split_features_target,
    validate_schema,
)
from bank_marketing.preprocessing import build_preprocessor, transformed_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing only.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_path = resolve_path(config["data"]["raw_path"])
    processed_dir = resolve_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_bank_data(raw_path)
    validate_schema(df)
    X, y = split_features_target(df, config["data"]["target_column"])
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    preprocessor = build_preprocessor(
        drop_duration=config["features"]["drop_duration_for_production"],
        scale_numeric=config["features"]["scale_numeric"],
    )
    preprocessor.fit(X_train, y_train)

    train_processed = transformed_dataframe(preprocessor, X_train, index=X_train.index)
    test_processed = transformed_dataframe(preprocessor, X_test, index=X_test.index)
    train_processed[config["data"]["target_column"]] = y_train
    test_processed[config["data"]["target_column"]] = y_test

    train_processed.to_csv(processed_dir / "train_processed.csv", index=False)
    test_processed.to_csv(processed_dir / "test_processed.csv", index=False)
    joblib.dump(preprocessor, processed_dir / "preprocessor.joblib")

    print(f"Saved processed train data: {processed_dir / 'train_processed.csv'}")
    print(f"Saved processed test data: {processed_dir / 'test_processed.csv'}")
    print(f"Saved fitted preprocessor: {processed_dir / 'preprocessor.joblib'}")


if __name__ == "__main__":
    main()
