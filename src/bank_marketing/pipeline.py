from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

from bank_marketing.models import build_model
from bank_marketing.preprocessing import build_preprocessor


def build_training_pipeline(
    model_name: str,
    random_state: int,
    drop_duration: bool,
    scale_numeric: bool,
    use_smote: bool,
    smote_sampling_strategy: float,
    scale_pos_weight: float | None,
    model_params: dict[str, Any] | None = None,
):
    preprocessor = build_preprocessor(
        drop_duration=drop_duration,
        scale_numeric=scale_numeric,
    )
    steps = list(preprocessor.steps)

    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=random_state,
                    sampling_strategy=smote_sampling_strategy,
                ),
            )
        )

    steps.append(
        (
            "classifier",
            build_model(
                model_name,
                random_state=random_state,
                scale_pos_weight=scale_pos_weight if not use_smote else None,
                params=model_params,
            ),
        )
    )

    pipeline_cls = ImbPipeline if use_smote else Pipeline
    return pipeline_cls(steps=steps)
