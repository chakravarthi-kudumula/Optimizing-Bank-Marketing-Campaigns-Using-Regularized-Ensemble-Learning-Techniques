# Bank Marketing Campaign Prediction

This project turns the original notebook analysis into a modular machine learning project for predicting whether a bank client will subscribe to a term deposit.

## Project Layout

```text
data/raw/                 Original CSV data
data/processed/           Output from the preprocessing pipeline
notebooks/                Preserved original notebook
src/bank_marketing/       Reusable package code
pipelines/                Runnable preprocessing, training, and prediction pipelines
artifacts/                Saved models, metrics, plots, and predictions
tests/                    Basic validation tests
config/config.yaml        Pipeline configuration
```

## Setup

```bash
python -m venv marketing
source marketing/bin/activate
pip install -r requirements.txt
```

## Run Preprocessing Only

This fits preprocessing on the training split, transforms train/test data, and saves inspectable CSVs.

```bash
python pipelines/preprocess_pipeline.py
```

Outputs:

```text
data/processed/train_processed.csv
data/processed/test_processed.csv
data/processed/preprocessor.joblib
```

## Train Model

```bash
python pipelines/train_pipeline.py
```

Outputs:

```text
artifacts/models/model.joblib
artifacts/metrics/metrics.json
artifacts/plots/confusion_matrix.png
artifacts/plots/roc_curve.png
artifacts/plots/feature_importance.png
```


## Compare Models

Use this first to reproduce the notebook's main model families as separate MLflow runs:

```bash
python pipelines/compare_models_pipeline.py
```

By default this runs the faster non-tuned variants:

```text
logistic_regression_unregularized
logistic_regression_l1
logistic_regression_l2
random_forest_baseline
xgboost_baseline
regularized_xgboost
```

To include RandomizedSearchCV tuned Random Forest and XGBoost runs too:

```bash
python pipelines/compare_models_pipeline.py --include-tuned
```

The comparison table is saved to:

```text
artifacts/metrics/model_comparison.csv
```

## Tune One Model

Use this when MLflow suggests a model is worth tuning more deeply:

```bash
python pipelines/tune_model_pipeline.py --model random_forest_tuned
python pipelines/tune_model_pipeline.py --model xgboost_tuned
```

Tuning settings live in `config/config.yaml`:

```yaml
tuning:
  n_iter: 10
  cv: 3
  scoring: roc_auc
```

## Promote The Best Model

After reviewing MLflow, update `config/config.yaml` instead of editing Python code. For example, if regularized XGBoost is best:

```yaml
model:
  name: regularized_xgboost
  use_smote: false
  smote_sampling_strategy: 0.5
  params: {}
```

If a tuned model wins, copy its best MLflow parameters into `model.params`, removing the `classifier__` prefix. Example:

```yaml
model:
  name: random_forest
  use_smote: true
  smote_sampling_strategy: 0.5
  params:
    n_estimators: 300
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: sqrt
    bootstrap: true
```

Then train the final production artifact:

```bash
python pipelines/train_pipeline.py
```


## MLflow Model Registry

Final training automatically registers the logged model when `registry.enabled` is true in `config/config.yaml`:

```yaml
registry:
  enabled: true
  registered_model_name: bank-marketing-subscription-model
  default_stage: candidate
```

Each registered version gets a lifecycle tag and alias. Supported stages are:

```text
candidate
staging
production
archived
```

After reviewing MLflow, promote a version like this:

```bash
python pipelines/set_model_stage_pipeline.py --version 1 --stage staging
python pipelines/set_model_stage_pipeline.py --version 1 --stage production
```

To archive a version:

```bash
python pipelines/set_model_stage_pipeline.py --version 1 --stage archived
```

The Registry view in MLflow will show the registered model and its versions. The aliases let you tell which version is the current candidate, staging model, or production model.

## MLflow Tracking

Training runs are logged to MLflow when `tracking.enabled` is true in `config/config.yaml`.

Install dependencies after this change:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python pipelines/train_pipeline.py
```

Start the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

MLflow will show parameters, metrics, saved plots, and the trained model for each run.

## Run Predictions

```bash
python pipelines/predict_pipeline.py --input data/raw/bank-full.csv --output artifacts/predictions.csv
```

## Production Note

The original notebook used `duration`, which is highly predictive but usually unavailable before a marketing call finishes. For a realistic pre-call production model, `config/config.yaml` sets:

```yaml
features:
  drop_duration_for_production: true
```

Set it to `false` only when reproducing notebook-style analysis where post-call information is allowed.

## Tests

```bash
pytest
```
