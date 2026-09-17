from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DEVELOPMENT_FILES = [
    "h20201332",
    "h20211332",
    "h20221331",
    "h20231331",
]

FINAL_TEST_FILE = "h20241331"


def _load_files(files: list[str]) -> pd.DataFrame:
    dfs = []

    for file in files:
        path = RAW_DATA_DIR / f"{file}data.csv"

        df = pd.read_csv(path)
        df["file"] = file

        dfs.append(df)

    return pd.concat(
        dfs,
        ignore_index=True,
    )


def load_development_data() -> pd.DataFrame:
    """Load the 2020-2023 development dataset."""
    return _load_files(DEVELOPMENT_FILES)


def load_final_test_data() -> pd.DataFrame:
    """Load the sealed 2024 final test dataset."""
    return _load_files([FINAL_TEST_FILE])