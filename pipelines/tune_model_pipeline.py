from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from bank_marketing.experiments import get_experiment, list_experiment_names
from bank_marketing.pipeline import build_training_pipeline
from bank_marketing.tracking import log_training_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune one model and log it to MLflow.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True, choices=list_experiment_names())
    args = parser.parse_args()

    config = load_config(args.config)
    experiment = get_experiment(args.model)
    if not experiment.search_params:
        raise ValueError(f"{experiment.name} does not define tuning search parameters.")

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
        model_name=experiment.model_name,
        random_state=config["data"]["random_state"],
        drop_duration=config["features"]["drop_duration_for_production"],
        scale_numeric=config["features"]["scale_numeric"],
        use_smote=experiment.use_smote,
        smote_sampling_strategy=config["model"]["smote_sampling_strategy"],
        scale_pos_weight=scale_pos_weight,
        model_params=experiment.model_params,
    )

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
    best_model = search.best_estimator_
    metrics = evaluate_classifier(best_model, X_train, y_train, X_test, y_test)

    metrics_root = resolve_path(config["artifacts"]["metrics_path"]).parent
    plots_dir = resolve_path(config["artifacts"]["plots_dir"]) / "tuning" / experiment.name
    metrics_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_root / f"{experiment.name}_tuning_metrics.json"
    confusion_matrix_path = plots_dir / "confusion_matrix.png"
    roc_curve_path = plots_dir / "roc_curve.png"
    feature_importance_path = plots_dir / "feature_importance.png"
    save_metrics(metrics, metrics_path)
    save_confusion_matrix(best_model, X_test, y_test, confusion_matrix_path)
    save_roc_curve(best_model, X_test, y_test, roc_curve_path)
    save_feature_importance(best_model, feature_importance_path)

    extra_params = {
        "run_type": "hyperparameter_tuning",
        "experiment_name": experiment.name,
        "model_name": experiment.model_name,
        "use_smote": experiment.use_smote,
        "search_type": "RandomizedSearchCV",
        "cv": config["tuning"]["cv"],
        "n_iter": config["tuning"]["n_iter"],
        "scoring": config["tuning"]["scoring"],
        "best_cv_score": float(search.best_score_),
    }
    extra_params.update({f"best__{k}": v for k, v in search.best_params_.items()})

    log_training_run(
        config={**config, "model": {**config["model"], "name": experiment.name, "use_smote": experiment.use_smote}},
        model=best_model,
        metrics=metrics,
        artifact_paths=[metrics_path, confusion_matrix_path, roc_curve_path, feature_importance_path],
        run_name=f"{experiment.name}_tuned",
        extra_params=extra_params,
    )

    print("Best parameters:")
    print(search.best_params_)
    print(f"Best CV {config['tuning']['scoring']}: {search.best_score_:.3f}")
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
