from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bank_marketing.config import load_config, resolve_path
from bank_marketing.data import (
    load_bank_data,
    make_train_test_split,
    split_features_target,
    validate_schema,
)
from bank_marketing.evaluate import (
    evaluate_classifier,
    save_confusion_matrix,
    save_feature_importance,
    save_metrics,
    save_roc_curve,
)
from bank_marketing.pipeline import build_training_pipeline
from bank_marketing.tracking import log_training_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bank marketing model.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_bank_data(resolve_path(config["data"]["raw_path"]))
    validate_schema(df)
    X, y = split_features_target(df, config["data"]["target_column"])
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    model = build_training_pipeline(
        model_name=config["model"]["name"],
        random_state=config["data"]["random_state"],
        drop_duration=config["features"]["drop_duration_for_production"],
        scale_numeric=config["features"]["scale_numeric"],
        use_smote=config["model"]["use_smote"],
        smote_sampling_strategy=config["model"]["smote_sampling_strategy"],
        scale_pos_weight=scale_pos_weight,
        model_params=config["model"].get("params"),
    )

    model.fit(X_train, y_train)
    metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test)

    model_path = resolve_path(config["artifacts"]["model_path"])
    metrics_path = resolve_path(config["artifacts"]["metrics_path"])
    plots_dir = resolve_path(config["artifacts"]["plots_dir"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    save_metrics(metrics, metrics_path)
    confusion_matrix_path = plots_dir / "confusion_matrix.png"
    roc_curve_path = plots_dir / "roc_curve.png"
    feature_importance_path = plots_dir / "feature_importance.png"
    save_confusion_matrix(model, X_test, y_test, confusion_matrix_path)
    save_roc_curve(model, X_test, y_test, roc_curve_path)
    save_feature_importance(model, feature_importance_path)

    log_training_run(
        config=config,
        model=model,
        metrics=metrics,
        artifact_paths=[
            model_path,
            metrics_path,
            confusion_matrix_path,
            roc_curve_path,
            feature_importance_path,
        ],
    )

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    if config.get("tracking", {}).get("enabled", False):
        print(f"Logged MLflow run to: {config['tracking']['tracking_uri']}")
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
