from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IRIS_CANDIDATES = [
    ROOT / "iris_big.csv",
    ROOT / "iris_big 1.csv",
]


def load_iris_df() -> pd.DataFrame:
    for path in IRIS_CANDIDATES:
        if path.exists():
            return cast(pd.DataFrame, pd.read_csv(path))
    raise FileNotFoundError("Nie znaleziono pliku iris_big.csv ani iris_big 1.csv")


def load_diagnosis_df() -> pd.DataFrame:
    path = ROOT / "diagnosis.csv"
    if not path.exists():
        raise FileNotFoundError("Nie znaleziono pliku diagnosis.csv")
    return cast(pd.DataFrame, pd.read_csv(path))


def split_xy(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.7,
    random_state: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = cast(pd.DataFrame, df.drop(columns=[target_col]))
    y = cast(pd.Series, df.loc[:, target_col])
    result = train_test_split(
        x, y, train_size=train_size, random_state=random_state, stratify=y
    )
    return cast(tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], result)
