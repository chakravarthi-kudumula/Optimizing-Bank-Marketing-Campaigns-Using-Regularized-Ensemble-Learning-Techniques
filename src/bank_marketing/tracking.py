from __future__ import annotations

from pathlib import Path
from typing import Any


VALID_REGISTRY_STAGES = {"candidate", "staging", "production", "archived"}


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for split_name, split_metrics in metrics.items():
        for metric_name, value in split_metrics.items():
            if isinstance(value, (int, float)):
                flattened[f"{split_name}_{metric_name}"] = float(value)
    return flattened


def setup_mlflow(config: dict[str, Any]):
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow tracking is enabled but mlflow is not installed. "
            "Run `pip install -r requirements.txt`."
        ) from exc

    tracking_config = config.get("tracking", {})
    mlflow.set_tracking_uri(tracking_config.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(tracking_config.get("experiment_name", "bank-marketing"))
    return mlflow


def normalize_stage(stage: str) -> str:
    normalized = stage.lower()
    if normalized not in VALID_REGISTRY_STAGES:
        raise ValueError(
            f"Invalid registry stage: {stage}. "
            f"Choose one of: {sorted(VALID_REGISTRY_STAGES)}"
        )
    return normalized


def log_training_run(
    config: dict[str, Any],
    model,
    metrics: dict[str, Any],
    artifact_paths: list[Path],
    run_name: str | None = None,
    extra_params: dict[str, Any] | None = None,
    log_model: bool = True,
) -> dict[str, Any] | None:
    tracking_config = config.get("tracking", {})
    if not tracking_config.get("enabled", False):
        return None

    mlflow = setup_mlflow(config)
    import mlflow.sklearn

    resolved_run_name = run_name or tracking_config.get("run_name") or config["model"]["name"]
    result: dict[str, Any] = {}

    with mlflow.start_run(run_name=resolved_run_name) as run:
        params = {
            "model_name": config["model"]["name"],
            "use_smote": config["model"].get("use_smote", False),
            "smote_sampling_strategy": config["model"].get("smote_sampling_strategy"),
            "drop_duration_for_production": config["features"][
                "drop_duration_for_production"
            ],
            "scale_numeric": config["features"]["scale_numeric"],
            "test_size": config["data"]["test_size"],
            "random_state": config["data"]["random_state"],
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params({key: value for key, value in params.items() if value is not None})
        mlflow.log_metrics(flatten_metrics(metrics))

        for path in artifact_paths:
            if path.exists():
                mlflow.log_artifact(str(path))

        result["run_id"] = run.info.run_id
        if log_model:
            model_info = mlflow.sklearn.log_model(model, name="model")
            result["model_uri"] = model_info.model_uri
            registered = maybe_register_model(
                config=config,
                model_uri=model_info.model_uri,
                run_id=run.info.run_id,
                metrics=metrics,
                extra_params=params,
            )
            if registered:
                result.update(registered)

    return result


def maybe_register_model(
    config: dict[str, Any],
    model_uri: str,
    run_id: str,
    metrics: dict[str, Any],
    extra_params: dict[str, Any],
) -> dict[str, Any] | None:
    registry_config = config.get("registry", {})
    if not registry_config.get("enabled", False):
        return None

    import mlflow
    from mlflow.tracking import MlflowClient

    registered_model_name = registry_config.get("registered_model_name")
    if not registered_model_name:
        raise ValueError("registry.registered_model_name must be configured.")

    stage = normalize_stage(registry_config.get("default_stage", "candidate"))
    registered_model = mlflow.register_model(model_uri, registered_model_name)
    version = str(registered_model.version)

    client = MlflowClient()
    client.set_registered_model_tag(
        registered_model_name,
        "project",
        registry_config.get("project_tag", "bank-marketing-campaign"),
    )
    client.set_model_version_tag(registered_model_name, version, "stage", stage)
    client.set_model_version_tag(registered_model_name, version, "source_run_id", run_id)
    client.set_model_version_tag(
        registered_model_name,
        version,
        "model_name",
        str(extra_params.get("model_name", config["model"]["name"])),
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "test_roc_auc",
        str(metrics["test"]["roc_auc"]),
    )
    client.set_registered_model_alias(registered_model_name, stage, version)

    return {
        "registered_model_name": registered_model_name,
        "registered_model_version": version,
        "registry_stage": stage,
    }


def set_model_stage(
    config: dict[str, Any],
    registered_model_name: str,
    version: str,
    stage: str,
) -> None:
    setup_mlflow(config)
    from mlflow.tracking import MlflowClient

    normalized_stage = normalize_stage(stage)
    client = MlflowClient()
    for alias in VALID_REGISTRY_STAGES:
        try:
            alias_version = client.get_model_version_by_alias(registered_model_name, alias)
        except Exception:
            continue
        if str(alias_version.version) == str(version) and alias != normalized_stage:
            client.delete_registered_model_alias(registered_model_name, alias)

    client.set_model_version_tag(registered_model_name, version, "stage", normalized_stage)
    client.set_registered_model_alias(registered_model_name, normalized_stage, version)
