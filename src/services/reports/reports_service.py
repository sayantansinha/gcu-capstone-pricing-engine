# src/services/pricing_reports_service.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------
@dataclass
class PortfolioOverview:
    overall_summary: Dict[str, Any]
    by_title: pd.DataFrame
    by_genre: Optional[pd.DataFrame]


@dataclass
class TerritoryIntelligence:
    by_territory: pd.DataFrame


@dataclass
class PlatformStrategy:
    by_platform: pd.DataFrame


@dataclass
class BundleValueReport:
    bundles: pd.DataFrame
    summary: Dict[str, Any]


@dataclass
class ExecutiveSummary:
    top_titles: pd.DataFrame
    top_territories: pd.DataFrame
    top_platforms: pd.DataFrame


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


def _safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


# ---------------------------------------------------------------------
# 1. Portfolio Pricing Overview
# ---------------------------------------------------------------------
def build_portfolio_overview(
        pricing_df: pd.DataFrame,
        top_n: int = 20,
) -> PortfolioOverview:
    """
    Aggregates value at portfolio, title, and genre levels.

    Works with the prediction outputs from the Price Predictor module.

    Expected columns (if present):
        - title_id, title_name
        - genre
        - predicted_price
    """
    if pricing_df.empty:
        empty = pd.DataFrame()
        return PortfolioOverview(
            overall_summary={
                "num_rows": 0,
                "num_titles": 0,
                "total_predicted_value": 0.0,
                "avg_predicted_price": 0.0,
                "median_predicted_price": 0.0,
            },
            by_title=empty,
            by_genre=None,
        )

    df = pricing_df.copy()
    df["predicted_price"] = _safe_float_series(df, "predicted_price")

    # Overall summary
    group_title_cols = [c for c in ["title_id", "title_name"] if c in df.columns]
    if group_title_cols:
        num_titles = df[group_title_cols].drop_duplicates().shape[0]
    else:
        num_titles = 0

    overall_summary = {
        "num_rows": int(df.shape[0]),
        "num_titles": int(num_titles),
        "total_predicted_value": float(df["predicted_price"].sum(skipna=True)),
        "avg_predicted_price": float(df["predicted_price"].mean(skipna=True)),
        "median_predicted_price": float(df["predicted_price"].median(skipna=True)),
    }

    # Per-title aggregation
    if group_title_cols:
        by_title = (
            df.groupby(group_title_cols, dropna=False)["predicted_price"]
            .agg(
                num_rows="count",
                total_predicted="sum",
                avg_predicted="mean",
                min_predicted="min",
                max_predicted="max",
            )
            .reset_index()
            .sort_values("total_predicted", ascending=False)
        )
        by_title_top = by_title.head(top_n).reset_index(drop=True)
    else:
        by_title_top = pd.DataFrame()

    # Per-genre aggregation (if available)
    if "genre" in df.columns:
        by_genre = (
            df.groupby("genre", dropna=False)["predicted_price"]
            .agg(
                num_titles="count",
                total_predicted="sum",
                avg_predicted="mean",
            )
            .reset_index()
            .sort_values("total_predicted", ascending=False)
        )
    else:
        by_genre = None

    return PortfolioOverview(
        overall_summary=overall_summary,
        by_title=by_title_top,
        by_genre=by_genre,
    )


# ---------------------------------------------------------------------
# 2. Territory Intelligence
# ---------------------------------------------------------------------
def build_territory_intelligence(
        pricing_df: pd.DataFrame,
) -> TerritoryIntelligence:
    """
    Summarizes predicted pricing behavior across territories.

    Works with the prediction outputs from Price Predictor.

    Expected columns:
        - territory
        - predicted_price
        - genre (optional)
    """
    if pricing_df.empty or "territory" not in pricing_df.columns:
        return TerritoryIntelligence(by_territory=pd.DataFrame())

    df = pricing_df.copy()
    df["predicted_price"] = _safe_float_series(df, "predicted_price")

    agg = (
        df.groupby("territory", dropna=False)["predicted_price"]
        .agg(
            num_rows="count",
            total_predicted="sum",
            avg_predicted="mean",
            median_predicted="median",
        )
        .reset_index()
    )

    # Optional: attach top genre per territory
    if "genre" in df.columns:
        genre_counts = (
            df.groupby(["territory", "genre"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        idx = genre_counts.groupby("territory")["count"].idxmax()
        top_genre = genre_counts.loc[idx, ["territory", "genre"]].rename(
            columns={"genre": "top_genre"}
        )
        agg = agg.merge(top_genre, on="territory", how="left")

    agg = agg.sort_values("total_predicted", ascending=False).reset_index(drop=True)
    return TerritoryIntelligence(by_territory=agg)


# ---------------------------------------------------------------------
# 3. Platform Strategy
# ---------------------------------------------------------------------
def build_platform_strategy(
        pricing_df: pd.DataFrame,
) -> PlatformStrategy:
    """
    Summarizes predicted pricing behavior across platforms (SVOD/TVOD/AVOD/etc).

    Works with Price Predictor outputs.

    Expected columns:
        - platform
        - predicted_price
        - genre (optional)
    """
    if pricing_df.empty or "platform" not in pricing_df.columns:
        return PlatformStrategy(by_platform=pd.DataFrame())

    df = pricing_df.copy()
    df["predicted_price"] = _safe_float_series(df, "predicted_price")

    agg = (
        df.groupby("platform", dropna=False)["predicted_price"]
        .agg(
            num_rows="count",
            total_predicted="sum",
            avg_predicted="mean",
            median_predicted="median",
        )
        .reset_index()
    )

    # Optional: attach top genre per platform
    if "genre" in df.columns:
        genre_counts = (
            df.groupby(["platform", "genre"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        idx = genre_counts.groupby("platform")["count"].idxmax()
        top_genre = genre_counts.loc[idx, ["platform", "genre"]].rename(
            columns={"genre": "top_genre"}
        )
        agg = agg.merge(top_genre, on="platform", how="left")

    agg = agg.sort_values("total_predicted", ascending=False).reset_index(drop=True)
    return PlatformStrategy(by_platform=agg)


# ---------------------------------------------------------------------
# 4. Bundle Value Optimization
# ---------------------------------------------------------------------
def _detect_bundle_group_cols(df: pd.DataFrame) -> Optional[List[str]]:
    """
    Try to detect columns to group bundles by, based on common naming patterns.

    We don't assume a specific schema; instead we look for:
        - 'bundle_id', 'bundle_label', or 'bundle_name'
    """
    candidates = []
    for col in ["bundle_id", "bundle_label", "bundle_name"]:
        if col in df.columns:
            candidates.append(col)
    return candidates or None


def _detect_bundle_value_col(df: pd.DataFrame) -> Optional[str]:
    """
    Detect a numeric 'value' column for bundles: first numeric column
    whose name contains 'price' or 'value'.
    """
    for col in df.columns:
        if "price" in col.lower() or "value" in col.lower():
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def build_bundle_value_report(
        bundling_df: pd.DataFrame,
) -> BundleValueReport:
    """
    Summarises bundle-level information coming from the Compare & Bundling module.

    Expected (flexible):
        - Some identifier columns (bundle_id / bundle_label / bundle_name)
        - A numeric total price / value column (contains 'price' or 'value' in its name)
        - Optional: 'num_titles' (otherwise inferred from group size)
    """
    if bundling_df is None or bundling_df.empty:
        return BundleValueReport(
            bundles=pd.DataFrame(),
            summary={"num_bundles": 0, "total_bundle_value": 0.0, "avg_bundle_value": 0.0},
        )

    df = bundling_df.copy()

    group_cols = _detect_bundle_group_cols(df)
    value_col = _detect_bundle_value_col(df)

    if not group_cols or not value_col:
        # We can't compute a meaningful bundle report with no identifiers or value column
        return BundleValueReport(
            bundles=pd.DataFrame(),
            summary={"num_bundles": 0, "total_bundle_value": 0.0, "avg_bundle_value": 0.0},
        )

    df[value_col] = _safe_float_series(df, value_col)

    if "num_titles" in df.columns:
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(
                num_titles=("num_titles", "max"),
                total_value=(value_col, "sum"),
                avg_value=(value_col, "mean"),
            )
            .reset_index()
        )
    else:
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(
                num_titles=("bundle_item_title" if "bundle_item_title" in df.columns else df.columns[0], "count"),
                total_value=(value_col, "sum"),
                avg_value=(value_col, "mean"),
            )
            .reset_index()
        )

    agg = agg.sort_values("total_value", ascending=False).reset_index(drop=True)

    summary = {
        "num_bundles": int(agg.shape[0]),
        "total_bundle_value": float(agg["total_value"].sum(skipna=True)),
        "avg_bundle_value": float(agg["total_value"].mean(skipna=True)),
    }

    return BundleValueReport(bundles=agg, summary=summary)


# ---------------------------------------------------------------------
# 5. Executive Pricing Intelligence Summary
# ---------------------------------------------------------------------
def build_executive_summary(
        pricing_df: pd.DataFrame,
        top_n_titles: int = 10,
        top_n_geo: int = 5,
        top_n_platforms: int = 5,
) -> ExecutiveSummary:
    """
    Builds a high-level summary that backs the Executive Pricing Intelligence tab.

    Uses:
        - Portfolio overview (top titles by predicted value)
        - Territory intelligence (top territories by total predicted value)
        - Platform strategy (top platforms by total predicted value)

    NOTE: We intentionally do NOT rely on 'historical_price' here, because
    Milestone 3 outputs are prediction-driven only.
    """
    po = build_portfolio_overview(pricing_df, top_n=top_n_titles)
    ti = build_territory_intelligence(pricing_df)
    ps = build_platform_strategy(pricing_df)

    top_titles = po.by_title.head(top_n_titles).reset_index(drop=True)

    if not ti.by_territory.empty:
        top_territories = ti.by_territory.head(top_n_geo).reset_index(drop=True)
    else:
        top_territories = pd.DataFrame()

    if not ps.by_platform.empty:
        top_platforms = ps.by_platform.head(top_n_platforms).reset_index(drop=True)
    else:
        top_platforms = pd.DataFrame()

    return ExecutiveSummary(
        top_titles=top_titles,
        top_territories=top_territories,
        top_platforms=top_platforms,
    )
