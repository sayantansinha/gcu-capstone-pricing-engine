from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import base64
import io

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

from src.utils.model_io_utils import (
    load_model_csv,
    load_model_json,
    model_run_exists,
)
from src.utils.data_io_utils import save_report_pdf
from src.utils.explain_utils import permutation_importance_scores, shap_summary_df
from src.services.pipeline.analytics.visual_tools_service import (
    chart_actual_vs_pred,
    chart_residuals,
    chart_residuals_qq,
)


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


@dataclass
class ModelAnalyticsReport:
    run_id: str
    per_model_metrics: pd.DataFrame
    ensemble_avg_metrics: Dict[str, Any]
    ensemble_weighted_metrics: Dict[str, Any]
    stacked_meta: Dict[str, Any]
    predictions: pd.DataFrame
    derived_metrics: Dict[str, Optional[float]]


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
    candidates = []
    for col in ["bundle_id", "bundle_label", "bundle_name"]:
        if col in df.columns:
            candidates.append(col)
    return candidates or None


def _detect_bundle_value_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if "price" in col.lower() or "value" in col.lower():
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
    return None


def build_bundle_value_report(
        bundling_df: pd.DataFrame,
) -> BundleValueReport:
    if bundling_df is None or bundling_df.empty:
        return BundleValueReport(
            bundles=pd.DataFrame(),
            summary={"num_bundles": 0, "total_bundle_value": 0.0, "avg_bundle_value": 0.0},
        )

    df = bundling_df.copy()

    group_cols = _detect_bundle_group_cols(df)
    value_col = _detect_bundle_value_col(df)

    if not group_cols or not value_col:
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
                num_titles=(
                    "bundle_item_title" if "bundle_item_title" in df.columns else df.columns[0],
                    "count",
                ),
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


# ---------------------------------------------------------------------
# 6. Model Analytics – data loading
# ---------------------------------------------------------------------
def _compute_derived_metrics_from_predictions(
        pred_df: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    if pred_df is None or pred_df.empty:
        return {}

    if not _has_columns(pred_df, ["y_true", "y_pred"]):
        return {}

    df = pred_df.copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        return {}

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    err = y_pred - y_true

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None

    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(np.abs(err[non_zero_mask] / y_true[non_zero_mask])) * 100.0
        )
    else:
        mape = None

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
    }


def build_model_analytics(run_id: str) -> ModelAnalyticsReport:
    run_id = run_id.strip()

    if not model_run_exists(run_id):
        empty_df = pd.DataFrame()
        return ModelAnalyticsReport(
            run_id=run_id,
            per_model_metrics=empty_df,
            ensemble_avg_metrics={},
            ensemble_weighted_metrics={},
            stacked_meta={},
            predictions=empty_df,
            derived_metrics={},
        )

    per_model_metrics = load_model_csv(run_id, "per_model_metrics.csv")
    if per_model_metrics is None:
        per_model_metrics = pd.DataFrame()

    predictions = load_model_csv(run_id, "predictions.csv")
    if predictions is None:
        predictions = pd.DataFrame()

    ensemble_avg = load_model_json(run_id, "ensemble_avg.json") or {}
    ensemble_wgt = load_model_json(run_id, "ensemble_weighted.json") or {}
    stacked_meta = load_model_json(run_id, "ensemble_stacked.json") or {}

    avg_metrics = ensemble_avg.get("metrics", {}) if isinstance(ensemble_avg, dict) else {}
    wgt_metrics = (
        ensemble_wgt.get("metrics", {}) if isinstance(ensemble_wgt, dict) else {}
    )

    derived_metrics = _compute_derived_metrics_from_predictions(predictions)

    return ModelAnalyticsReport(
        run_id=run_id,
        per_model_metrics=per_model_metrics,
        ensemble_avg_metrics=avg_metrics,
        ensemble_weighted_metrics=wgt_metrics,
        stacked_meta=stacked_meta if isinstance(stacked_meta, dict) else {},
        predictions=predictions,
        derived_metrics=derived_metrics,
    )


# ---------------------------------------------------------------------
# 7. Model Analytics – PDF generation with BP + SHAP + PI
# ---------------------------------------------------------------------
def _line_chart_y_true_pred(y_true, y_pred, max_width: float) -> Optional[Image]:
    """
    Build a simple line chart for y_true vs y_pred (index on x-axis)
    and return as a ReportLab Image, scaled to max_width.
    """
    if y_true is None or y_pred is None or len(y_true) == 0:
        return None

    try:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(len(y_true)), y_true, label="y_true")
        ax.plot(range(len(y_pred)), y_pred, label="y_pred")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)

        img = Image(buf)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_width / float(iw), 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        return img
    except Exception:
        return None


def _image_from_data_uri(data_uri: str, max_width: float) -> Optional[Image]:
    """Convert a base64 data URI from visual_tools_service into a ReportLab Image."""
    if not data_uri:
        return None
    try:
        if "," in data_uri:
            _, b64 = data_uri.split(",", 1)
        else:
            b64 = data_uri
        img_bytes = base64.b64decode(b64)
        buf = io.BytesIO(img_bytes)
        img = Image(buf)
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_width / float(iw), 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        return img
    except Exception:
        return None


def _metrics_dict_to_table_data(metrics: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = [["Metric", "Value"]]
    if not metrics:
        return rows
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            val = f"{v:.4f}"
        else:
            val = str(v)
        rows.append([str(k), val])
    return rows


def generate_model_analytics_report_pdf(
        run_id: str,
        report: "ModelAnalyticsReport",  # type: ignore[name-defined]
        model_results: Optional[Dict[str, Any]] = None,
        report_name: str = "model_analytics_report",
        model_name: Optional[str] = None,
) -> str:
    """
    Generate a Model Analytics PDF that mirrors the Model Analytics tab:
      - Overall diagnostics (RMSE/MAE/R²/MAPE)
      - Ensemble metrics (avg / weighted / stacked) as tables
      - Per-model metrics table (BP JSON column stripped out)
      - Breusch–Pagan test table (parsed from per_model_metrics last column or last_model)
      - Permutation importance (validation set) table
      - SHAP summary (mean |SHAP|) table
      - Prediction samples (y_true vs y_pred) table
      - Residual diagnostics charts (Actual vs Pred, Residuals vs Pred, Q–Q)
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    story: List[Any] = []

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    story.append(Paragraph("Model Analytics Report", styles["Title"]))
    story.append(Spacer(1, 12))

    # Prefer the human-friendly model name; fall back to run_id if missing
    label = model_name or run_id
    story.append(Paragraph(f"Model: {label}", styles["Normal"]))
    generated_str = datetime.now().strftime("%b %d, %Y %I:%M %p")
    story.append(Paragraph(f"Generated: {generated_str}", styles["Normal"]))
    story.append(Spacer(1, 16))

    # ------------------------------------------------------------------
    # Overall diagnostics
    # ------------------------------------------------------------------
    if report.derived_metrics:
        story.append(Paragraph("Overall diagnostics (from predictions)", styles["Heading2"]))
        data = [["Metric", "Value"]]

        def _fmt(v: Optional[float], pct: bool = False) -> str:
            if v is None:
                return "n/a"
            return f"{v:.2f}%" if pct else f"{v:.4f}"

        data.append(["RMSE", _fmt(report.derived_metrics.get("rmse"))])
        data.append(["MAE", _fmt(report.derived_metrics.get("mae"))])
        data.append(["R²", _fmt(report.derived_metrics.get("r2"))])
        data.append(["MAPE", _fmt(report.derived_metrics.get("mape"), pct=True)])

        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 12))

    # ------------------------------------------------------------------
    # Ensemble metrics (tables, not JSON)
    # ------------------------------------------------------------------
    avg_metrics = report.ensemble_avg_metrics or {}
    wgt_metrics = report.ensemble_weighted_metrics or {}
    stack_metrics = (report.stacked_meta or {}).get("metrics") or {}

    if avg_metrics or wgt_metrics or stack_metrics:
        story.append(Paragraph("Ensemble metrics", styles["Heading2"]))

        # Average ensemble
        story.append(Paragraph("Average ensemble", styles["Heading3"]))
        avg_data = _metrics_dict_to_table_data(avg_metrics)
        avg_tbl = Table(avg_data, repeatRows=1)
        avg_tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(avg_tbl)
        story.append(Spacer(1, 8))

        # Weighted ensemble
        story.append(Paragraph("Weighted ensemble (1/RMSE)", styles["Heading3"]))
        wgt_data = _metrics_dict_to_table_data(wgt_metrics)
        wgt_tbl = Table(wgt_data, repeatRows=1)
        wgt_tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(wgt_tbl)
        story.append(Spacer(1, 8))

        # Stacked ensemble
        story.append(Paragraph("Stacked ensemble (hybrid)", styles["Heading3"]))
        stack_data = _metrics_dict_to_table_data(stack_metrics)
        stack_tbl = Table(stack_data, repeatRows=1)
        stack_tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(stack_tbl)
        story.append(Spacer(1, 12))

    # ------------------------------------------------------------------
    # Per-model metrics (strip BP JSON column) + capture BP dict
    # ------------------------------------------------------------------
    bp_from_per_model: Optional[Dict[str, Any]] = None

    if not report.per_model_metrics.empty:
        df = report.per_model_metrics.copy()
        last_col = df.columns[-1]
        series = df[last_col]

        for cell in series.dropna().tolist():
            if isinstance(cell, dict):
                bp_from_per_model = cell
                break
            text = str(cell).strip()
            if not text:
                continue
            if text.startswith("{") and text.endswith("}"):
                try:
                    bp_from_per_model = json.loads(text)
                    break
                except Exception:
                    try:
                        parsed = ast.literal_eval(text)
                        if isinstance(parsed, dict):
                            bp_from_per_model = parsed
                            break
                    except Exception:
                        continue

        if isinstance(bp_from_per_model, dict):
            df = df.drop(columns=[last_col])

        story.append(Paragraph("Per-model metrics", styles["Heading2"]))
        data = [list(df.columns)] + df.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 12))

    # ------------------------------------------------------------------
    # BP table (same precedence as UI: per_model -> last_model)
    # ------------------------------------------------------------------
    mr = model_results or {}
    bp_dict: Optional[Dict[str, Any]] = None

    if isinstance(bp_from_per_model, dict) and bp_from_per_model:
        bp_dict = bp_from_per_model
    else:
        if isinstance(mr, dict):
            bp_dict = mr.get("bp_results") or mr.get("bp")

    if isinstance(bp_dict, dict) and bp_dict:
        story.append(Paragraph("Breusch–Pagan test (heteroscedasticity)", styles["Heading2"]))
        data = [["Statistic", "Value"]]
        for k, v in bp_dict.items():
            data.append([str(k), str(v)])
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 12))

    # ------------------------------------------------------------------
    # Permutation importance & SHAP (same as UI)
    # ------------------------------------------------------------------
    model = mr.get("model") if isinstance(mr, dict) else None
    X_valid = mr.get("X_valid") if isinstance(mr, dict) else None
    y_valid = mr.get("y_valid") if isinstance(mr, dict) else None
    X_sample = mr.get("X_sample") if isinstance(mr, dict) else None

    # Permutation importance
    pi_df = None
    if model is not None and X_valid is not None and y_valid is not None:
        try:
            pi_df = permutation_importance_scores(model, X_valid, y_valid, n_repeats=5)
        except Exception:
            pi_df = None

    if pi_df is not None and not pi_df.empty:
        story.append(Paragraph("Permutation importance (validation set)", styles["Heading2"]))
        df_pi = pi_df.copy().reset_index(drop=True).head(25)
        data = [list(df_pi.columns)] + df_pi.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 12))

    # SHAP summary
    shap_df = None
    if model is not None and X_sample is not None:
        try:
            shap_df = shap_summary_df(model, X_sample)
        except Exception:
            shap_df = None

    if shap_df is not None and not shap_df.empty:
        story.append(Paragraph("SHAP summary (mean |SHAP|)", styles["Heading2"]))
        df_sh = shap_df.copy().reset_index(drop=True).head(25)
        data = [list(df_sh.columns)] + df_sh.astype(str).values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 12))

        # ------------------------------------------------------------------
    # Prediction samples (y_true vs y_pred) + y_true/y_pred line chart
    # + residual diagnostics charts
    # ------------------------------------------------------------------
    preds = report.predictions
    if not preds.empty and {"y_true", "y_pred"}.issubset(preds.columns):
        y_true = pd.to_numeric(preds["y_true"], errors="coerce")
        y_pred = pd.to_numeric(preds["y_pred"], errors="coerce")
        mask = y_true.notna() & y_pred.notna()
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if not y_true.empty:
            # 1) Table of y_true / y_pred (like UI top of diagnostics)
            story.append(Paragraph("Prediction samples (y_true vs y_pred)", styles["Heading2"]))
            sample_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).head(50)
            data = [list(sample_df.columns)] + sample_df.astype(str).values.tolist()
            tbl = Table(data, repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 12))

            # 2) y_true vs y_pred line chart (to match the Streamlit st.line_chart)
            line_img = _line_chart_y_true_pred(y_true, y_pred, max_width=doc.width)
            if line_img is not None:
                story.append(Paragraph("y_true vs y_pred (line chart)", styles["Heading3"]))
                story.append(line_img)
                story.append(Spacer(1, 12))

            # 3) Residual charts (same three as UI)
            story.append(Paragraph("Residual diagnostics", styles["Heading2"]))

            uri1 = chart_actual_vs_pred(y_true, y_pred)
            uri2 = chart_residuals(y_true, y_pred)
            uri3 = chart_residuals_qq(y_true, y_pred)

            for title, uri in [
                ("Actual vs Predicted", uri1),
                ("Residuals vs Predicted", uri2),
                ("Residuals Q–Q plot", uri3),
            ]:
                img = _image_from_data_uri(uri, max_width=doc.width)
                if img is None:
                    continue
                story.append(Paragraph(title, styles["Heading3"]))
                story.append(img)
                story.append(Spacer(1, 12))

    # ------------------------------------------------------------------
    # Footer with page numbers + save
    # ------------------------------------------------------------------
    def _numbered(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(doc_.pagesize[0] - 40, 25, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_numbered, onLaterPages=_numbered)
    pdf_bytes = buf.getvalue()
    buf.close()

    base_dir = run_id or "model_analytics"
    ref = save_report_pdf(base_dir=base_dir, name=report_name, pdf_bytes=pdf_bytes)
    return ref
