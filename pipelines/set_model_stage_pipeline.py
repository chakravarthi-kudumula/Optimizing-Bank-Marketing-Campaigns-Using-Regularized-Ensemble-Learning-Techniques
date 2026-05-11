from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bank_marketing.config import load_config
from bank_marketing.tracking import VALID_REGISTRY_STAGES, set_model_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Set an MLflow Model Registry stage alias.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--version", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(VALID_REGISTRY_STAGES))
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = args.model_name or config["registry"]["registered_model_name"]
    set_model_stage(
        config=config,
        registered_model_name=model_name,
        version=args.version,
        stage=args.stage,
    )
    print(f"Set {model_name} version {args.version} to {args.stage}")


if __name__ == "__main__":
    main()
