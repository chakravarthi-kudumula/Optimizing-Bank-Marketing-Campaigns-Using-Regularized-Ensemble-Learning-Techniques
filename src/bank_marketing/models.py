from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _merge_params(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    params = defaults.copy()
    if overrides:
        params.update(overrides)
    return params


def build_model(
    name: str,
    random_state: int,
    scale_pos_weight: float | None = None,
    params: dict[str, Any] | None = None,
):
    normalized = name.lower().replace("_", "-")

    if normalized in {"logistic", "logistic-regression", "logistic-regression-l2"}:
        return LogisticRegression(
            **_merge_params(
                {
                    "penalty": "l2",
                    "max_iter": 2000,
                    "solver": "lbfgs",
                    "class_weight": "balanced",
                    "random_state": random_state,
                },
                params,
            )
        )

    if normalized in {"logistic-regression-none", "logistic-regression-unregularized"}:
        return LogisticRegression(
            **_merge_params(
                {
                    "penalty": None,
                    "max_iter": 2000,
                    "solver": "lbfgs",
                    "class_weight": "balanced",
                    "random_state": random_state,
                },
                params,
            )
        )

    if normalized in {"logistic-regression-l1", "logistic-l1"}:
        return LogisticRegression(
            **_merge_params(
                {
                    "penalty": "l1",
                    "max_iter": 3000,
                    "solver": "saga",
                    "class_weight": "balanced",
                    "random_state": random_state,
                },
                params,
            )
        )

    if normalized in {"random-forest", "rf", "random-forest-baseline"}:
        return RandomForestClassifier(
            **_merge_params(
                {
                    "n_estimators": 200,
                    "max_depth": None,
                    "class_weight": "balanced",
                    "random_state": random_state,
                    "n_jobs": -1,
                },
                params,
            )
        )

    if normalized in {"xgboost", "xgb", "regularized-xgboost"}:
        return _build_xgboost(
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            regularized=True,
            params=params,
        )

    if normalized in {"xgboost-baseline", "xgb-baseline"}:
        return _build_xgboost(
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            regularized=False,
            params=params,
        )

    raise ValueError(f"Unsupported model name: {name}")


def _build_xgboost(
    random_state: int,
    scale_pos_weight: float | None,
    regularized: bool,
    params: dict[str, Any] | None,
):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for XGBoost models. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    defaults: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "eval_metric": "logloss",
    }

    if regularized:
        defaults.update(
            {
                "n_estimators": 400,
                "min_child_weight": 4,
                "gamma": 0.2,
                "reg_lambda": 2,
                "reg_alpha": 1,
            }
        )

    if scale_pos_weight is not None:
        defaults["scale_pos_weight"] = scale_pos_weight

    return XGBClassifier(**_merge_params(defaults, params))
