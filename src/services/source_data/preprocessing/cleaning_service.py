from typing import Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def impute(df: pd.DataFrame, strategy: str = "median", columns: Optional[List[str]] = None) -> pd.DataFrame:
    cols = columns or df.columns.tolist()
    out = df.copy()
    for c in cols:
        if out[c].dtype.kind in "biufc":
            if strategy == "median":
                out[c] = out[c].fillna(out[c].median())
            elif strategy == "mean":
                out[c] = out[c].fillna(out[c].mean())
        else:
            out[c] = out[c].fillna(out[c].mode().iloc[0] if out[c].mode().size else out[c])
    return out


def iqr_filter(df: pd.DataFrame, column: str, k: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    q1, q3 = out[column].quantile(0.25), out[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return out[(out[column] >= lower) & (out[column] <= upper)]


def winsorize(df: pd.DataFrame, column: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    lo, hi = out[column].quantile(lower_q), out[column].quantile(upper_q)
    out[column] = out[column].clip(lo, hi)
    return out


def encode_one_hot(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def encode_ordinal(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    for col, order in mapping.items():
        out[col] = pd.Categorical(out[col], categories=order, ordered=True).codes
    return out


def scale(df: pd.DataFrame, columns: List[str], method: str = "standard") -> pd.DataFrame:
    out = df.copy()
    if method == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    out[columns] = scaler.fit_transform(out[columns])
    return out


def deduplicate(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)
