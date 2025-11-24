from typing import Dict, List
import numpy as np
import pandas as pd


# ---------- Helper for safe datetime handling ----------

def _as_datetime_series(obj, index=None) -> pd.Series:
    """
    Ensure consistent datetime Series for .dt access.
    Handles both Series and DatetimeIndex safely.
    """
    if isinstance(obj, pd.DatetimeIndex):
        return obj.to_series(index=index)
    if isinstance(obj, pd.Series):
        return pd.to_datetime(obj, errors="coerce")
    return pd.to_datetime(pd.Series(obj), errors="coerce")


# ---------- Datetime EDA helpers ----------

def _datetime_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()


def datetime_basic_summary(df: pd.DataFrame) -> Dict:
    """Summarize datetime fields without numeric conversion."""
    summaries: Dict[str, Dict] = {}
    for col in _datetime_columns(df):
        s = _as_datetime_series(df[col]).dropna()
        if s.empty:
            summaries[col] = {
                "count": 0,
                "min": None,
                "max": None,
                "range_days": None,
                "missing": int(df[col].isna().sum()),
            }
            continue
        mn, mx = s.min(), s.max()
        rng = (mx - mn).days if hasattr(mx - mn, "days") else None
        summaries[col] = {
            "count": int(s.size),
            "min": mn,
            "max": mx,
            "range_days": rng,
            "missing": int(df[col].isna().sum()),
        }
    return summaries


def datetime_trend_counts(df: pd.DataFrame, col: str, freq: str = "M") -> Dict[str, int]:
    """Counts rows by period (month, week, etc.) for a datetime column."""
    if col not in df.columns:
        return {}
    ser = _as_datetime_series(df[col]).dropna()
    if ser.empty:
        return {}
    counts = ser.dt.to_period(freq).value_counts().sort_index()
    counts.index = counts.index.to_timestamp()
    return {ts.isoformat(): int(cnt) for ts, cnt in counts.items()}


# ---------- Main EDA summary ----------

def eda_summary(df: pd.DataFrame) -> Dict:
    """
    Generates comprehensive EDA for pandas ≥ 2.x
    - Handles datetimes natively (no conversion)
    - Computes numeric correlations
    - Returns summary stats, missingness, dtypes, samples, and time summaries
    """
    n_rows, n_cols = df.shape
    types = df.dtypes.astype(str).to_dict()
    missing = df.isna().sum().astype(int).to_dict()

    # General describe (pandas 2.x handles datetime)
    try:
        desc = df.describe(include="all").to_dict()
    except Exception:
        desc = {c: df[c].describe().to_dict() for c in df.columns}

    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr(numeric_only=True).round(3).to_dict() if not num_df.empty else {}
    sample = df.head(10).to_dict(orient="records")

    # Datetime section
    dt_summary = datetime_basic_summary(df)
    dt_trends = {col: datetime_trend_counts(df, col, "M") for col in _datetime_columns(df)}

    return {
        "shape": {"rows": int(n_rows), "cols": int(n_cols)},
        "dtypes": types,
        "missing": missing,
        "describe": desc,
        "corr_numeric": corr,
        "sample_head": sample,
        "datetime_summary": dt_summary,
        "datetime_trends_monthly": dt_trends,
    }
