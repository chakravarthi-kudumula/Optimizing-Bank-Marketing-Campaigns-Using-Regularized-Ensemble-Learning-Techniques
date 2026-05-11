from pathlib import Path

from bank_marketing.data import EXPECTED_COLUMNS, load_bank_data, validate_schema


def test_raw_data_schema():
    data_path = Path("data/raw/bank-full.csv")
    df = load_bank_data(data_path)
    validate_schema(df)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 45211
