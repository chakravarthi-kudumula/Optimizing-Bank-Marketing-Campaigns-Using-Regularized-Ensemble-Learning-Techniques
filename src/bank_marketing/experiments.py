from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelExperiment:
    name: str
    model_name: str
    use_smote: bool = False
    model_params: dict[str, Any] = field(default_factory=dict)
    tuned: bool = False
    search_params: dict[str, list[Any]] = field(default_factory=dict)


MODEL_EXPERIMENTS: list[ModelExperiment] = [
    ModelExperiment(
        name="logistic_regression_unregularized",
        model_name="logistic_regression_none",
        use_smote=True,
    ),
    ModelExperiment(
        name="logistic_regression_l2",
        model_name="logistic_regression_l2",
        use_smote=True,
    ),
    ModelExperiment(
        name="logistic_regression_l1",
        model_name="logistic_regression_l1",
        use_smote=True,
    ),
    ModelExperiment(
        name="random_forest_baseline",
        model_name="random_forest",
        use_smote=True,
    ),
    ModelExperiment(
        name="random_forest_tuned",
        model_name="random_forest",
        use_smote=True,
        tuned=True,
        search_params={
            "classifier__n_estimators": [100, 200, 300, 400],
            "classifier__max_depth": [None, 5, 10, 15, 20],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
            "classifier__bootstrap": [True, False],
        },
    ),
    ModelExperiment(
        name="xgboost_baseline",
        model_name="xgboost_baseline",
        use_smote=True,
    ),
    ModelExperiment(
        name="xgboost_tuned",
        model_name="xgboost_baseline",
        use_smote=True,
        tuned=True,
        search_params={
            "classifier__n_estimators": [100, 200, 300, 400],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__max_depth": [3, 4, 5, 6, 8],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__gamma": [0, 0.1, 0.3, 0.5],
            "classifier__reg_lambda": [0.5, 1, 1.5, 2.0],
        },
    ),
    ModelExperiment(
        name="regularized_xgboost",
        model_name="regularized_xgboost",
        use_smote=False,
    ),
]


def get_experiment(name: str) -> ModelExperiment:
    for experiment in MODEL_EXPERIMENTS:
        if experiment.name == name or experiment.model_name == name:
            return experiment
    raise ValueError(f"Unknown experiment/model: {name}")


def list_experiment_names() -> list[str]:
    return [experiment.name for experiment in MODEL_EXPERIMENTS]
