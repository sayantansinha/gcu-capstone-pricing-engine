from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data_io_utils import save_figure


# ---------- Helper ----------

def _as_datetime_series(obj, index=None) -> pd.Series:
    """Ensure consistent datetime Series for .dt access."""
    if isinstance(obj, pd.DatetimeIndex):
        return obj.to_series(index=index)
    if isinstance(obj, pd.Series):
        return pd.to_datetime(obj, errors="coerce")
    return pd.to_datetime(pd.Series(obj), errors="coerce")


def _fig_name(run_id: str, name: str) -> str:
    return f"{run_id}_{name}"


# ---------- Numeric & Categorical plots ----------

def plot_hist(df: pd.DataFrame, column: str, run_id: str, bins: int = 30) -> str:
    ser = df[column].dropna()
    fig, ax = plt.subplots()
    ser.plot(kind="hist", bins=bins, ax=ax)
    ax.set_title(f"Histogram: {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    out = save_figure(fig, run_id, _fig_name(run_id, f"hist_{column}"))
    plt.close(fig)
    return out


def plot_box(df: pd.DataFrame, column: str, run_id: str) -> str:
    ser = df[column].dropna()
    fig, ax = plt.subplots()
    ser.plot(kind="box", ax=ax)
    ax.set_title(f"Boxplot: {column}")
    out = save_figure(fig, run_id, _fig_name(run_id, f"box_{column}"))
    plt.close(fig)
    return out


def plot_bar(df: pd.DataFrame, cat: str, val: Optional[str], run_id: str, topn: int = 20) -> str:
    if val is None:
        vc = df[cat].value_counts().head(topn)
        title = f"Bar: {cat} (count)"
    else:
        vc = df.groupby(cat, dropna=False)[val].mean().sort_values(ascending=False).head(topn)
        title = f"Bar: {cat} (mean {val})"
    fig, ax = plt.subplots()
    vc.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(cat)
    out = save_figure(fig, run_id, _fig_name(run_id, f"bar_{cat}{'_' + val if val else ''}"))
    plt.close(fig)
    return out


def plot_scatter(df: pd.DataFrame, x: str, y: str, run_id: str) -> str:
    mask = df[x].notna() & df[y].notna()
    fig, ax = plt.subplots()
    ax.scatter(df.loc[mask, x], df.loc[mask, y])
    ax.set_title(f"Scatter: {x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    out = save_figure(fig, run_id, _fig_name(run_id, f"scatter_{x}_{y}"))
    plt.close(fig)
    return out


# ---------- Datetime visualizations ----------

def plot_datetime_counts(df: pd.DataFrame, datetime_col: str, run_id: str, freq: str = "M") -> str:
    """Line plot showing frequency of records by time period."""
    ser = _as_datetime_series(df[datetime_col]).dropna()
    if ser.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"No data to plot for {datetime_col} ({freq})")
        out = save_figure(fig, run_id, _fig_name(run_id, f"dt_counts_{datetime_col}_{freq}"))
        plt.close(fig)
        return out

    counts = ser.dt.to_period(freq).value_counts().sort_index()
    counts.index = counts.index.to_timestamp()
    fig, ax = plt.subplots()
    counts.plot(kind="line", ax=ax)
    ax.set_title(f"{datetime_col} frequency over time ({freq})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    out = save_figure(fig, run_id, _fig_name(run_id, f"dt_counts_{datetime_col}_{freq}"))
    plt.close(fig)
    return out


def plot_time_of_day_hist(df: pd.DataFrame, datetime_col: str, run_id: str) -> str:
    """Histogram of hours (0–23) if datetime column includes time-of-day."""
    ser = _as_datetime_series(df[datetime_col]).dropna()
    if ser.empty:
        fig, ax = plt.subplots()
        ax.set_title(f"No hour-of-day data for {datetime_col}")
        out = save_figure(fig, run_id, _fig_name(run_id, f"dt_hour_{datetime_col}"))
        plt.close(fig)
        return out

    hours = ser.dt.hour
    fig, ax = plt.subplots()
    hours.plot(kind="hist", bins=24, ax=ax)
    ax.set_title(f"Distribution by hour: {datetime_col}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Count")
    out = save_figure(fig, run_id, _fig_name(run_id, f"dt_hour_{datetime_col}"))
    plt.close(fig)
    return out


def plot_month_box(df: pd.DataFrame, datetime_col: str, value_col: str, run_id: str) -> str:
    """Boxplot of numeric value by month derived from a datetime column."""
    ser = _as_datetime_series(df[datetime_col])
    valid = df[value_col].notna() & ser.notna()
    if not valid.any():
        fig, ax = plt.subplots()
        ax.set_title(f"No data for {value_col} by month ({datetime_col})")
        out = save_figure(fig, run_id, _fig_name(run_id, f"dt_month_box_{value_col}_by_{datetime_col}"))
        plt.close(fig)
        return out

    tmp = pd.DataFrame({
        value_col: df.loc[valid, value_col],
        "month": ser.loc[valid].dt.month
    })
    fig, ax = plt.subplots()
    tmp.boxplot(column=value_col, by="month", ax=ax)
    ax.set_title(f"{value_col} by month")
    ax.set_xlabel("Month")
    ax.set_ylabel(value_col)
    fig.suptitle("")  # remove default matplotlib title
    out = save_figure(fig, run_id, _fig_name(run_id, f"dt_month_box_{value_col}_by_{datetime_col}"))
    plt.close(fig)
    return out
