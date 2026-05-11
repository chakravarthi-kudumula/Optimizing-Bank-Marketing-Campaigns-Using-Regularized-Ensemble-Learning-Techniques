from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

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
from bank_marketing.experiments import MODEL_EXPERIMENTS
from bank_marketing.pipeline import build_training_pipeline
from bank_marketing.tracking import log_training_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare notebook model variants with MLflow.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--include-tuned", action="store_true", help="Include RandomizedSearchCV experiments.")
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
    plots_root = resolve_path(config["artifacts"]["plots_dir"]) / "model_comparison"
    metrics_root = resolve_path(config["artifacts"]["metrics_path"]).parent
    plots_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    comparison_rows = []
    for experiment in MODEL_EXPERIMENTS:
        if experiment.tuned and not args.include_tuned:
            continue

        print(f"Training {experiment.name}...")
        model = build_training_pipeline(
            model_name=experiment.model_name,
            random_state=config["data"]["random_state"],
            drop_duration=config["features"]["drop_duration_for_production"],
            scale_numeric=config["features"]["scale_numeric"],
            use_smote=experiment.use_smote,
            smote_sampling_strategy=config["model"]["smote_sampling_strategy"],
            scale_pos_weight=scale_pos_weight,
            model_params=experiment.model_params,
        )

        extra_params = {
            "run_type": "model_comparison",
            "experiment_name": experiment.name,
            "model_name": experiment.model_name,
            "is_tuned": experiment.tuned,
            "use_smote": experiment.use_smote,
        }

        if experiment.tuned:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=experiment.search_params,
                n_iter=config["tuning"]["n_iter"],
                scoring=config["tuning"]["scoring"],
                cv=config["tuning"]["cv"],
                verbose=1,
                random_state=config["data"]["random_state"],
                n_jobs=config["tuning"]["n_jobs"],
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            extra_params.update({f"best__{k}": v for k, v in search.best_params_.items()})
            extra_params["best_cv_score"] = float(search.best_score_)
        else:
            model.fit(X_train, y_train)

        metrics = evaluate_classifier(model, X_train, y_train, X_test, y_test)
        run_plot_dir = plots_root / experiment.name
        run_plot_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_root / f"{experiment.name}_metrics.json"
        confusion_matrix_path = run_plot_dir / "confusion_matrix.png"
        roc_curve_path = run_plot_dir / "roc_curve.png"
        feature_importance_path = run_plot_dir / "feature_importance.png"

        save_metrics(metrics, metrics_path)
        save_confusion_matrix(model, X_test, y_test, confusion_matrix_path)
        save_roc_curve(model, X_test, y_test, roc_curve_path)
        save_feature_importance(model, feature_importance_path)

        log_training_run(
            config={**config, "model": {**config["model"], "name": experiment.name, "use_smote": experiment.use_smote}},
            model=model,
            metrics=metrics,
            artifact_paths=[metrics_path, confusion_matrix_path, roc_curve_path, feature_importance_path],
            run_name=experiment.name,
            extra_params=extra_params,
        )

        comparison_rows.append(
            {
                "experiment": experiment.name,
                "model_name": experiment.model_name,
                "is_tuned": experiment.tuned,
                "use_smote": experiment.use_smote,
                "train_roc_auc": metrics["train"]["roc_auc"],
                "test_roc_auc": metrics["test"]["roc_auc"],
                "test_accuracy": metrics["test"]["accuracy"],
                "test_precision": metrics["test"]["precision"],
                "test_recall": metrics["test"]["recall"],
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("test_roc_auc", ascending=False)
    output_path = metrics_root / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(f"Saved comparison table: {output_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
