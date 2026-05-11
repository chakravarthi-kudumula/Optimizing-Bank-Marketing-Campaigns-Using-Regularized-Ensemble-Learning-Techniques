from pathlib import Path

from bank_marketing.data import load_bank_data, split_features_target, validate_schema
from bank_marketing.preprocessing import build_preprocessor


def test_preprocessor_transforms_sample_without_duration():
    df = load_bank_data(Path("data/raw/bank-full.csv")).head(50)
    validate_schema(df)
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(drop_duration=True)
    transformed = preprocessor.fit_transform(X, y)
    feature_names = preprocessor.get_feature_names_out()
    assert transformed.shape[0] == 50
    assert "duration" not in feature_names
    assert "duration_log" not in feature_names
