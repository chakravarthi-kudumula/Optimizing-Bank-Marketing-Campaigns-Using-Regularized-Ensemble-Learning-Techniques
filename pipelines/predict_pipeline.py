from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bank_marketing.config import load_config, resolve_path
from bank_marketing.data import EXPECTED_COLUMNS, load_bank_data, validate_schema
from bank_marketing.inference import load_model, predict_dataframe


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict term deposit subscription.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default="artifacts/predictions.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = resolve_path(args.input or config["data"]["raw_path"])
    model_path = resolve_path(config["artifacts"]["model_path"])
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_bank_data(input_path)
    if config["data"]["target_column"] in df.columns:
        validate_schema(df)
        X = df.drop(columns=[config["data"]["target_column"]])
    else:
        expected_without_target = set(EXPECTED_COLUMNS) - {config["data"]["target_column"]}
        missing = sorted(expected_without_target - set(df.columns))
        if missing:
            raise ValueError(f"Missing required input columns: {missing}")
        X = df

    model = load_model(model_path)
    predictions = predict_dataframe(model, X)
    predictions.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")


if __name__ == "__main__":
    main()
